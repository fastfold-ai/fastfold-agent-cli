"""Versioned API models for the FastFold agent server."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Capabilities(ApiModel):
    api_version: Literal["v1"] = "v1"
    sessions: bool = True
    event_replay: bool = True
    local_filesystem: bool = False
    skills: bool = False
    integrations: bool = False
    mcp: bool = False
    terminal: bool = False
    execution_backends: list[str] = Field(default_factory=lambda: ["local"])


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"] = "ok"
    service: str = "FastFold Agent Server"
    version: str
    backend_id: str
    user_name: str | None = None
    capabilities: Capabilities = Field(default_factory=Capabilities)


SessionStatus = Literal["idle", "running", "interrupted", "error"]
OrganizeLabel = Literal["pinned", "archive"]


class AgentProject(ApiModel):
    id: str
    name: str
    description: str | None = None
    agent_context: str | None = None
    pinned: bool = False
    session_count: int = 0
    created_at: datetime
    updated_at: datetime


class CreateProjectRequest(ApiModel):
    name: str = Field(min_length=1)
    description: str | None = None
    agent_context: str | None = None
    pinned: bool = False


class UpdateProjectRequest(ApiModel):
    name: str | None = None
    description: str | None = None
    agent_context: str | None = None
    pinned: bool | None = None
    clear_description: bool = False
    clear_agent_context: bool = False


class ProjectList(ApiModel):
    data: list[AgentProject]


class DeleteProjectResponse(ApiModel):
    deleted: bool = True


class AgentSession(ApiModel):
    id: str
    title: str
    status: SessionStatus = "idle"
    organize_label: OrganizeLabel | None = None
    project_id: str | None = None
    workspace_path: str | None = None
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CreateSessionRequest(ApiModel):
    title: str | None = None
    workspace_path: str | None = None
    project_id: str | None = None


class UpdateSessionRequest(ApiModel):
    title: str | None = None
    organize_label: OrganizeLabel | None = None
    clear_organize_label: bool = False
    project_id: str | None = None
    clear_project_id: bool = False


class AutoTitleSessionRequest(ApiModel):
    force: bool = False


MessageRole = Literal["user", "assistant", "system"]


class AgentMessage(ApiModel):
    id: str
    session_id: str
    role: MessageRole
    content: str
    created_at: datetime


class SendMessageRequest(ApiModel):
    content: str = Field(min_length=1)


class StartRunResponse(ApiModel):
    run_id: str


class EventEnvelope(ApiModel):
    schema_version: Literal[1] = 1
    id: str
    session_id: str
    run_id: str | None = None
    sequence: int
    timestamp: datetime
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionList(ApiModel):
    data: list[AgentSession]


class MessageList(ApiModel):
    data: list[AgentMessage]
    feedback: dict[str, Literal["up", "down"]] = Field(default_factory=dict)


class DeleteSessionResponse(ApiModel):
    deleted: bool = True


class InterruptRunResponse(ApiModel):
    interrupted: bool


class WorkspaceFile(ApiModel):
    path: str
    name: str
    type: Literal["file", "directory"]
    size: int | None = None
    mtime: datetime | None = None


class WorkspaceFileList(ApiModel):
    data: list[WorkspaceFile]


class WorkspaceFileContent(ApiModel):
    path: str
    content: str
    encoding: Literal["text", "base64"] = "text"
    mime_type: str
    version: str
    size: int
    mtime: datetime


class WriteWorkspaceFileRequest(ApiModel):
    path: str
    content: str
    encoding: Literal["text", "base64"] = "text"
    base_version: str | None = None


class CreateWorkspaceFolderRequest(ApiModel):
    path: str


class MoveWorkspaceFileRequest(ApiModel):
    source_path: str
    target_path: str


class WorkspaceMutationResponse(ApiModel):
    path: str


class SkillSummary(ApiModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    source: str
    author: str
    version: str | None = None
    updated_at: str | None = None


class SkillList(ApiModel):
    data: list[SkillSummary]


class SkillDetail(SkillSummary):
    content: str
    directory: str | None = None


class InstallSkillRequest(ApiModel):
    source: str = Field(min_length=1)


class SkillMutationResponse(ApiModel):
    ok: bool
    summary: str
    installed: list[str] = Field(default_factory=list)


class IntegrationField(ApiModel):
    env_var: str
    label: str
    is_secret: bool = True
    configured: bool
    source: Literal["environment", "config", "none"]
    masked_preview: str | None = None


class IntegrationProvider(ApiModel):
    key: str
    name: str
    category: str
    description: str
    env_var: str
    configured: bool
    source: Literal["environment", "config", "none"]
    masked_preview: str | None = None
    setup_url: str | None = None
    free: bool = False
    fields: list[IntegrationField] = Field(default_factory=list)


class IntegrationList(ApiModel):
    data: list[IntegrationProvider]


class UpdateIntegrationRequest(ApiModel):
    value: str | None = None
    values: dict[str, str] = Field(default_factory=dict)


class ValidateIntegrationResponse(ApiModel):
    ok: bool
    configured: bool
    source: Literal["environment", "config", "none"]
    message: str


class RuntimeSettings(ApiModel):
    provider: str
    model: str
    openai_base_url: str | None = None
    agent_profile: str
    tool_mode: str


class UpdateRuntimeSettingsRequest(ApiModel):
    provider: str | None = None
    model: str | None = None
    openai_base_url: str | None = None
    agent_profile: str | None = None
    tool_mode: str | None = None


class McpServer(ApiModel):
    id: str
    name: str
    transport: Literal["stdio", "sse", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    enabled: bool = True
    created_at: datetime
    updated_at: datetime


class McpServerList(ApiModel):
    data: list[McpServer]


class CreateMcpServerRequest(ApiModel):
    name: str = Field(min_length=1)
    transport: Literal["stdio", "sse", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    enabled: bool = True


class UpdateMcpServerRequest(ApiModel):
    name: str | None = None
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    enabled: bool | None = None


class MessageFeedbackRequest(ApiModel):
    reaction: Literal["up", "down"] | None = None


class MessageFeedbackResponse(ApiModel):
    message_id: str
    reaction: Literal["up", "down"] | None = None


class PtySessionInfo(ApiModel):
    id: str
    session_id: str
    title: str
    cwd: str
    cols: int = 80
    rows: int = 24
    pid: int
    status: Literal["running", "exited"] = "running"


class PtySessionList(ApiModel):
    data: list[PtySessionInfo]


class CreatePtyRequest(ApiModel):
    title: str | None = None
    cols: int = Field(default=80, ge=1, le=500)
    rows: int = Field(default=24, ge=1, le=200)


class UpdatePtyRequest(ApiModel):
    title: str | None = None
    cols: int | None = Field(default=None, ge=1, le=500)
    rows: int | None = Field(default=None, ge=1, le=200)
