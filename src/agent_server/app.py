"""FastAPI transport for the FastFold agent application service."""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from _version import __version__
from agent_server.models import (
    AgentSession,
    Capabilities,
    CreateSessionRequest,
    DeleteSessionResponse,
    HealthResponse,
    IntegrationList,
    IntegrationProvider,
    InterruptRunResponse,
    MessageList,
    MessageFeedbackRequest,
    MessageFeedbackResponse,
    McpServer,
    McpServerList,
    CreateMcpServerRequest,
    MoveWorkspaceFileRequest,
    RuntimeSettings,
    SendMessageRequest,
    SessionList,
    StartRunResponse,
    InstallSkillRequest,
    SkillDetail,
    SkillList,
    SkillMutationResponse,
    UpdateIntegrationRequest,
    UpdateRuntimeSettingsRequest,
    UpdateMcpServerRequest,
    ValidateIntegrationResponse,
    UpdateSessionRequest,
    AutoTitleSessionRequest,
    CreateWorkspaceFolderRequest,
    WorkspaceFileContent,
    WorkspaceFileList,
    WorkspaceMutationResponse,
    WriteWorkspaceFileRequest,
)
from agent_server.service import AgentService, SessionBusyError
from agent_server.skills_service import SkillsService
from agent_server.integrations_service import IntegrationsService
from agent_server.settings_service import SettingsService
from agent_server.store import AgentStore
from agent_server.workspace import (
    MAX_FILE_SIZE,
    WorkspaceConflictError,
    WorkspacePathError,
    WorkspaceService,
)

DEFAULT_LOCAL_ORIGINS = [
    "http://127.0.0.1:8969",
    "http://localhost:8969",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def local_user_name() -> str:
    try:
        import pwd

        full_name = pwd.getpwuid(os.getuid()).pw_gecos.split(",", 1)[0].strip()
        if full_name:
            return full_name.split()[0]
    except (ImportError, KeyError, OSError):
        pass
    return getpass.getuser().replace(".", " ").replace("_", " ").title()


def create_app(
    *,
    store_path: Path | None = None,
    api_key: str | None = None,
    allowed_origins: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    store = AgentStore(store_path)
    service = AgentService(store)
    skills_service = SkillsService()
    integrations_service = IntegrationsService()
    settings_service = SettingsService()
    backend_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(store.path.resolve())))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await service.shutdown()

    app = FastAPI(
        title="FastFold Agent Server",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.agent_service = service

    def session_workspace(session_id: str) -> WorkspaceService:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        try:
            session = service.ensure_workspace(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        if not session.workspace_path:
            raise HTTPException(status_code=409, detail="Session has no workspace.")
        try:
            return WorkspaceService(Path(session.workspace_path))
        except WorkspacePathError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def _workspace_http_error(exc: Exception) -> HTTPException:
        if isinstance(exc, FileNotFoundError):
            return HTTPException(status_code=404, detail="Workspace path not found.")
        if isinstance(exc, FileExistsError):
            return HTTPException(status_code=409, detail="Workspace target already exists.")
        if isinstance(exc, (IsADirectoryError, NotADirectoryError, WorkspacePathError)):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail="Workspace operation failed.")

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts or ["127.0.0.1", "localhost", "testserver"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or DEFAULT_LOCAL_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["authorization", "content-type", "last-event-id"],
    )

    if api_key:

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            if request.url.path.startswith("/v1/"):
                authorization = request.headers.get("authorization", "")
                scheme, _, candidate = authorization.partition(" ")
                valid = (
                    scheme.lower() == "bearer"
                    and bool(candidate)
                    and secrets.compare_digest(candidate, api_key)
                )
                if not valid:
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid backend API key."},
                    )
            return await call_next(request)

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            version=__version__,
            backend_id=backend_id,
            user_name=local_user_name(),
            capabilities=Capabilities(
                local_filesystem=True,
                skills=True,
                integrations=True,
                mcp=True,
            ),
        )

    @app.get("/v1/mcp-servers", response_model=McpServerList)
    async def list_mcp_servers() -> McpServerList:
        return McpServerList(data=await asyncio.to_thread(store.list_mcp_servers))

    @app.post("/v1/mcp-servers", response_model=McpServer, status_code=201)
    async def create_mcp_server(payload: CreateMcpServerRequest) -> McpServer:
        if payload.transport == "stdio" and not payload.command:
            raise HTTPException(status_code=400, detail="stdio MCP servers require a command.")
        if payload.transport != "stdio" and not payload.url:
            raise HTTPException(status_code=400, detail="Remote MCP servers require a URL.")
        return await asyncio.to_thread(
            store.create_mcp_server,
            name=payload.name,
            transport=payload.transport,
            command=payload.command,
            args=payload.args,
            url=payload.url,
            enabled=payload.enabled,
        )

    @app.patch("/v1/mcp-servers/{server_id}", response_model=McpServer)
    async def update_mcp_server(
        server_id: str,
        payload: UpdateMcpServerRequest,
    ) -> McpServer:
        server = await asyncio.to_thread(
            store.update_mcp_server,
            server_id,
            **payload.model_dump(exclude_unset=True),
        )
        if server is None:
            raise HTTPException(status_code=404, detail="MCP server not found.")
        return server

    @app.delete("/v1/mcp-servers/{server_id}", response_model=DeleteSessionResponse)
    async def delete_mcp_server(server_id: str) -> DeleteSessionResponse:
        if not await asyncio.to_thread(store.delete_mcp_server, server_id):
            raise HTTPException(status_code=404, detail="MCP server not found.")
        return DeleteSessionResponse()

    @app.get("/v1/integrations", response_model=IntegrationList)
    async def list_integrations() -> IntegrationList:
        return IntegrationList(data=await asyncio.to_thread(integrations_service.list))

    @app.get("/v1/integrations/{integration_key}", response_model=IntegrationProvider)
    async def get_integration(integration_key: str) -> IntegrationProvider:
        provider = await asyncio.to_thread(integrations_service.get, integration_key)
        if provider is None:
            raise HTTPException(status_code=404, detail="Integration not found.")
        return provider

    @app.put("/v1/integrations/{integration_key}", response_model=IntegrationProvider)
    async def update_integration(
        integration_key: str,
        payload: UpdateIntegrationRequest,
    ) -> IntegrationProvider:
        try:
            values = dict(payload.values)
            if payload.value:
                provider = await asyncio.to_thread(
                    integrations_service.get,
                    integration_key,
                )
                if provider is None:
                    raise KeyError(integration_key)
                values[provider.env_var] = payload.value
            return await asyncio.to_thread(
                integrations_service.update,
                integration_key,
                values,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Integration not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/integrations/{integration_key}", response_model=IntegrationProvider)
    async def remove_integration(integration_key: str) -> IntegrationProvider:
        try:
            return await asyncio.to_thread(integrations_service.remove, integration_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Integration not found.") from exc

    @app.post(
        "/v1/integrations/{integration_key}/validate",
        response_model=ValidateIntegrationResponse,
    )
    async def validate_integration(integration_key: str) -> ValidateIntegrationResponse:
        try:
            return await asyncio.to_thread(integrations_service.validate, integration_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Integration not found.") from exc

    @app.get("/v1/settings/runtime", response_model=RuntimeSettings)
    async def get_runtime_settings() -> RuntimeSettings:
        return await asyncio.to_thread(settings_service.get)

    @app.patch("/v1/settings/runtime", response_model=RuntimeSettings)
    async def update_runtime_settings(
        payload: UpdateRuntimeSettingsRequest,
    ) -> RuntimeSettings:
        try:
            return await asyncio.to_thread(settings_service.update, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/skills", response_model=SkillList)
    async def list_skills() -> SkillList:
        return SkillList(data=await asyncio.to_thread(skills_service.list))

    @app.get("/v1/skills/{skill_name}", response_model=SkillDetail)
    async def get_skill(skill_name: str) -> SkillDetail:
        skill = await asyncio.to_thread(skills_service.get, skill_name)
        if skill is None:
            raise HTTPException(status_code=404, detail="Skill not found.")
        return skill

    @app.post("/v1/skills", response_model=SkillMutationResponse)
    async def install_skill(payload: InstallSkillRequest) -> SkillMutationResponse:
        result = await asyncio.to_thread(skills_service.install, payload.source)
        if not result.ok:
            raise HTTPException(status_code=400, detail=result.summary)
        return result

    @app.delete("/v1/skills/{skill_name}", response_model=SkillMutationResponse)
    async def remove_skill(skill_name: str) -> SkillMutationResponse:
        result = await asyncio.to_thread(skills_service.remove, skill_name)
        if not result.ok:
            raise HTTPException(status_code=400, detail=result.summary)
        return result

    @app.post("/v1/skills/upgrade", response_model=SkillMutationResponse)
    async def upgrade_skills() -> SkillMutationResponse:
        return await asyncio.to_thread(skills_service.upgrade)

    @app.get("/v1/sessions", response_model=SessionList)
    async def list_sessions() -> SessionList:
        return SessionList(data=await asyncio.to_thread(store.list_sessions))

    @app.get("/v1/sessions/search", response_model=SessionList)
    async def search_sessions(
        q: str = Query(default="", max_length=200),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> SessionList:
        return SessionList(
            data=await asyncio.to_thread(store.search_sessions, q, limit=limit)
        )

    @app.post("/v1/sessions", response_model=AgentSession, status_code=201)
    async def create_session(payload: CreateSessionRequest) -> AgentSession:
        try:
            return await asyncio.to_thread(
                service.create_session,
                title=payload.title,
                workspace_path=payload.workspace_path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}", response_model=AgentSession)
    async def get_session(session_id: str) -> AgentSession:
        session = await asyncio.to_thread(store.get_session, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session

    @app.patch("/v1/sessions/{session_id}", response_model=AgentSession)
    async def update_session(
        session_id: str,
        payload: UpdateSessionRequest,
    ) -> AgentSession:
        update_label = payload.organize_label is not None or payload.clear_organize_label
        try:
            return await asyncio.to_thread(
                service.update_session,
                session_id=session_id,
                title=payload.title,
                organize_label=payload.organize_label,
                update_organize_label=update_label,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/title/auto", response_model=AgentSession)
    async def auto_title_session(
        session_id: str,
        payload: AutoTitleSessionRequest | None = None,
    ) -> AgentSession:
        body = payload or AutoTitleSessionRequest()
        try:
            return await asyncio.to_thread(
                service.auto_title_session,
                session_id,
                force=body.force,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc

    @app.delete("/v1/sessions/{session_id}", response_model=DeleteSessionResponse)
    async def delete_session(session_id: str) -> DeleteSessionResponse:
        if not await service.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        return DeleteSessionResponse()

    @app.get("/v1/sessions/{session_id}/messages", response_model=MessageList)
    async def list_messages(session_id: str) -> MessageList:
        if await asyncio.to_thread(store.get_session, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return MessageList(
            data=await asyncio.to_thread(store.list_messages, session_id),
            feedback=await asyncio.to_thread(store.get_session_feedback, session_id),
        )

    @app.post(
        "/v1/sessions/{session_id}/messages/{message_id}/feedback",
        response_model=MessageFeedbackResponse,
    )
    async def update_message_feedback(
        session_id: str,
        message_id: str,
        payload: MessageFeedbackRequest,
    ) -> MessageFeedbackResponse:
        if await asyncio.to_thread(store.get_session, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        saved = await asyncio.to_thread(
            store.set_message_feedback,
            message_id,
            payload.reaction,
        )
        if not saved:
            raise HTTPException(status_code=404, detail="Message not found.")
        return MessageFeedbackResponse(
            message_id=message_id,
            reaction=payload.reaction,
        )

    @app.post(
        "/v1/sessions/{session_id}/messages",
        response_model=StartRunResponse,
        status_code=202,
    )
    async def send_message(
        session_id: str,
        payload: SendMessageRequest,
    ) -> StartRunResponse:
        try:
            return service.submit_message(session_id=session_id, content=payload.content)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/sessions/{session_id}/interrupt",
        response_model=InterruptRunResponse,
    )
    async def interrupt_session(session_id: str) -> InterruptRunResponse:
        if await asyncio.to_thread(store.get_session, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return InterruptRunResponse(interrupted=service.interrupt(session_id))

    @app.get(
        "/v1/sessions/{session_id}/files",
        response_model=WorkspaceFileList,
    )
    async def list_workspace_files(
        session_id: str,
        path: str = Query(default=""),
        recursive: bool = Query(default=True),
    ) -> WorkspaceFileList:
        workspace = session_workspace(session_id)
        try:
            data = await asyncio.to_thread(workspace.list, path, recursive=recursive)
            return WorkspaceFileList(data=data)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.get(
        "/v1/sessions/{session_id}/files/content",
        response_model=WorkspaceFileContent,
    )
    async def read_workspace_file(
        session_id: str,
        path: str = Query(min_length=1),
    ) -> WorkspaceFileContent:
        workspace = session_workspace(session_id)
        try:
            return await asyncio.to_thread(workspace.read, path)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.put(
        "/v1/sessions/{session_id}/files/content",
        response_model=WorkspaceFileContent,
    )
    async def write_workspace_file(
        session_id: str,
        payload: WriteWorkspaceFileRequest,
    ):
        workspace = session_workspace(session_id)
        try:
            return await asyncio.to_thread(
                workspace.write,
                payload.path,
                content=payload.content,
                encoding=payload.encoding,
                base_version=payload.base_version,
            )
        except WorkspaceConflictError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(exc),
                    "current": exc.current.model_dump(mode="json", by_alias=True),
                },
            )
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/v1/sessions/{session_id}/files/download")
    async def download_workspace_file(
        session_id: str,
        path: str = Query(min_length=1),
    ) -> FileResponse:
        workspace = session_workspace(session_id)
        try:
            file_path = workspace.resolve(path, must_exist=True)
            if not file_path.is_file():
                raise IsADirectoryError(path)
            return FileResponse(file_path, filename=file_path.name)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.post(
        "/v1/sessions/{session_id}/files/upload",
        response_model=WorkspaceFileContent,
    )
    async def upload_workspace_file(
        session_id: str,
        file: UploadFile = File(...),
        relative_dir: str = Form(default="uploads"),
    ) -> WorkspaceFileContent:
        workspace = session_workspace(session_id)
        data = await file.read(MAX_FILE_SIZE + 1)
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File exceeds the 40MB limit.")
        relative_path = str(Path(relative_dir) / (file.filename or "upload.bin"))
        try:
            return await asyncio.to_thread(workspace.upload, relative_path, data)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.post(
        "/v1/sessions/{session_id}/files/folders",
        response_model=WorkspaceMutationResponse,
    )
    async def create_workspace_folder(
        session_id: str,
        payload: CreateWorkspaceFolderRequest,
    ) -> WorkspaceMutationResponse:
        workspace = session_workspace(session_id)
        try:
            path = await asyncio.to_thread(workspace.create_folder, payload.path)
            return WorkspaceMutationResponse(path=path)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.post(
        "/v1/sessions/{session_id}/files/move",
        response_model=WorkspaceMutationResponse,
    )
    async def move_workspace_file(
        session_id: str,
        payload: MoveWorkspaceFileRequest,
    ) -> WorkspaceMutationResponse:
        workspace = session_workspace(session_id)
        try:
            path = await asyncio.to_thread(
                workspace.move,
                payload.source_path,
                payload.target_path,
            )
            return WorkspaceMutationResponse(path=path)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.delete(
        "/v1/sessions/{session_id}/files",
        response_model=WorkspaceMutationResponse,
    )
    async def delete_workspace_file(
        session_id: str,
        path: str = Query(min_length=1),
        recursive: bool = Query(default=False),
    ) -> WorkspaceMutationResponse:
        workspace = session_workspace(session_id)
        try:
            await asyncio.to_thread(workspace.delete, path, recursive=recursive)
            return WorkspaceMutationResponse(path=path)
        except Exception as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/v1/sessions/{session_id}/events")
    async def stream_events(
        request: Request,
        session_id: str,
        after_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        if await asyncio.to_thread(store.get_session, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        header_sequence = request.headers.get("last-event-id", "").strip()
        cursor = after_sequence
        if header_sequence.isdigit():
            cursor = max(cursor, int(header_sequence))

        async def event_source() -> AsyncIterator[str]:
            nonlocal cursor
            heartbeat_ticks = 0
            while not await request.is_disconnected():
                events = await asyncio.to_thread(
                    store.list_events,
                    session_id,
                    after_sequence=cursor,
                )
                if events:
                    heartbeat_ticks = 0
                    for event in events:
                        cursor = event.sequence
                        data = json.dumps(
                            event.model_dump(mode="json", by_alias=True),
                            separators=(",", ":"),
                        )
                        yield f"id: {event.sequence}\nevent: {event.type}\ndata: {data}\n\n"
                else:
                    heartbeat_ticks += 1
                    if heartbeat_ticks >= 15:
                        heartbeat_ticks = 0
                        yield ": heartbeat\n\n"
                    await asyncio.sleep(1)

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app
