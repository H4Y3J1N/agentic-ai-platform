# 모듈식 Agentic AI 플랫폼 - 완전 구현 가이드 Part 1

> **버전**: 1.0.0  
> **최종 업데이트**: 2025-01-07  
> **핵심 컨셉**: "AI 엔진은 똑같은데, 도메인 지식만 갈아끼우면 되는 구조"

---

## 목차 (Part 1)

1. [비전 및 아키텍처 개요](#1-비전-및-아키텍처-개요)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [코어 컴포넌트 구현](#3-코어-컴포넌트-구현)
   - 3.1 [Orchestrator 시스템](#31-orchestrator-시스템)
   - 3.2 [Agent 시스템](#32-agent-시스템)
   - 3.3 [Tool 시스템](#33-tool-시스템)
4. [RAG 시스템](#4-rag-시스템)
5. [LLM Gateway](#5-llm-gateway)

---

## 1. 비전 및 아키텍처 개요

### 1.1 핵심 비전

```
[Agentic AI Platform - 재사용 가능한 템플릿]
         │
         ├─ 복제 → [회사: 농작업 AI Service]
         │           └─ 농업 도메인 RAG + 페르소나
         │
         └─ 복제 → [개인사업: 쇼핑몰 AI Service]
                     └─ 커머스 도메인 RAG + 페르소나
```

### 1.2 템플릿 기반 배포

```bash
# 새 도메인 적용 시
git clone agentic-ai-platform
cd agentic-ai-platform

# 설정 파일만 수정
vim config/domain.yaml
---
domain: "ecommerce"  # agriculture / manufacturing / ...
persona: "쇼핑몰 운영 전문가"
rag_source: "s3://ecommerce-docs"
agents:
  - customer_service
  - inventory_optimizer
  - marketing_analyst
---

# 배포
docker build -t ai-service-ecommerce .
docker run -p 8000:8000 ai-service-ecommerce
```

### 1.3 설계 원칙

| 원칙 | 설명 |
|------|------|
| **코어 재사용** | Orchestrator, RAG, LLM Gateway는 100% 재사용 |
| **도메인 분리** | Agent, Tool, Knowledge는 도메인별로 구현 |
| **설정 기반** | YAML 설정으로 동작 변경 (코드 수정 최소화) |
| **관찰 가능** | 모든 동작을 추적, 측정, 시각화 |

### 1.4 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   고객 앱    │  │  운영자 웹   │  │  관리자 API  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼────────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       API GATEWAY LAYER                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  FastAPI / HTTP Endpoints                                   │    │
│  │  - POST /agent/execute                                      │    │
│  │  - POST /agent/chat                                         │    │
│  │  - GET  /agent/status                                       │    │
│  └─────────────────────────┬──────────────────────────────────┘    │
└────────────────────────────┼───────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ JWT 인증     │→ │ RBAC/PBAC    │→ │ Rate Limiter │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                             ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Audit Logger (모든 요청/응답 기록)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼───────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                  Agent Orchestrator                         │    │
│  │  ┌──────────────────────────────────────────────────┐      │    │
│  │  │  Pattern Router (Supervisor/Hierarchy/etc)       │      │    │
│  │  └──────────────────────────────────────────────────┘      │    │
│  │                        ↓                                    │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │    │
│  │  │ Customer     │  │ Inventory    │  │ Marketing    │    │    │
│  │  │ Service      │  │ Manager      │  │ Analyst      │    │    │
│  │  │ Agent        │  │ Agent        │  │ Agent        │    │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │    │
│  └─────────┼──────────────────┼──────────────────┼───────────┘    │
└────────────┼──────────────────┼──────────────────┼────────────────┘
             │                  │                  │
             └──────────┬───────┴──────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TOOL LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ CustomerDB   │  │ OrderLookup  │  │ PaymentAPI   │              │
│  │ Tool         │  │ Tool         │  │ Tool         │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                  │                  │                      │
│         └──────────────────┴──────────────────┘                     │
│                             ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           Tool Security Wrapper (권한 검사)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┼───────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE LAYER                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    RAG Engine                               │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │    │
│  │  │   Embedder   │→ │ Vector Store │→ │  Retriever   │    │    │
│  │  │              │  │   (Milvus)   │  │              │    │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────┼───────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM LAYER                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    LLM Gateway                              │    │
│  │  ┌──────────────────────────────────────────────────┐      │    │
│  │  │  Router (cost/performance/hybrid)                │      │    │
│  │  └──────────────────────────────────────────────────┘      │    │
│  │                        ↓                                    │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │    │
│  │  │  OpenAI    │  │ Anthropic  │  │ Local vLLM │          │    │
│  │  │  Provider  │  │ Provider   │  │ / Ollama   │          │    │
│  │  └────────────┘  └────────────┘  └────────────┘          │    │
│  │                                                             │    │
│  │  ┌──────────────────────────────────────────────────────┐ │    │
│  │  │  Cache Layer (Redis / Local)                         │ │    │
│  │  └──────────────────────────────────────────────────────┘ │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────┼───────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ PostgreSQL   │  │   Milvus     │  │    Redis     │              │
│  │ (비즈니스)   │  │  (벡터DB)    │  │   (캐시)     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   MONITORING LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Langfuse    │  │ Prometheus   │  │   Jaeger     │              │
│  │ (LLM 추적)   │  │  (메트릭)    │  │  (트레이싱)  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 프로젝트 구조

```
agentic-ai-platform-complete/
│
├── .gitignore
├── .dockerignore   
├── BACKEND_FEATURES.md
├── ZERO_DOWNTIME_DEPLOYMENT.md
│
├── packages/
│   └── agentic-ai-core/
│       └── src/
│           └── agentic_ai_core/
│               ├── __init__.py
│               ├── __version__.py
│               │
│               ├── agents/
│               │   ├── __init__.py
│               │   ├── agent_registry.py
│               │   ├── base_agent.py
│               │   └── streaming_agent.py
│               │
│               ├── api/
│               │   ├── __init__.py
│               │   ├── base_router.py
│               │   ├── exception_handler.py
│               │   ├── pagination.py
│               │   ├── rate_limiter.py
│               │   ├── response_models.py
│               │   ├── sse_response.py
│               │   └── websocket_manager.py
│               │
│               ├── database/
│               │   ├── __init__.py
│               │   └── connection.py
│               │
│               ├── llm/
│               │   ├── __init__.py
│               │   ├── cache.py
│               │   ├── gateway.py
│               │   ├── router.py
│               │   ├── streaming_gateway.py
│               │   └── providers/
│               │       ├── __init__.py
│               │       ├── anthropic_provider.py
│               │       ├── base.py
│               │       └── openai_provider.py
│               │
│               ├── observability/
│               │   ├── __init__.py
│               │   ├── langfuse_client.py
│               │   └── tracer.py
│               │
│               ├── orchestrator/
│               │   ├── __init__.py
│               │   ├── base_orchestrator.py
│               │   ├── message_bus.py
│               │   ├── secure_orchestrator.py
│               │   ├── workflow_engine.py
│               │   └── patterns/
│               │       ├── __init__.py
│               │       ├── collaborative.py
│               │       ├── hierarchy.py
│               │       ├── sequential.py
│               │       └── supervisor.py
│               │
│               ├── rag/
│               │   ├── __init__.py
│               │   ├── chunker.py
│               │   ├── embedder.py
│               │   ├── indexer.py
│               │   ├── milvus_store.py
│               │   ├── query_rewriter.py
│               │   ├── retriever.py
│               │   └── vector_store.py
│               │
│               ├── security/
│               │   ├── __init__.py
│               │   ├── pbac_engine.py
│               │   ├── rbac.py
│               │   ├── auth/
│               │   │   ├── __init__.py
│               │   │   └── jwt_handler.py
│               │   └── rate_limiting/
│               │       ├── __init__.py
│               │       ├── decorators.py
│               │       ├── redis_limiter.py
│               │       ├── sliding_window.py
│               │       └── token_bucket.py
│               │
│               ├── tools/
│               │   ├── __init__.py
│               │   ├── base_tool.py
│               │   ├── tool_registry.py
│               │   └── tool_security_wrapper.py
│               │
│               └── utils/
│                   ├── __init__.py
│                   ├── config_loader.py
│                   └── logger.py
│
├── services/
│   └── ecommerce/
│       ├── .dockerignore  
│       ├── README.md
│       ├── pyproject.toml
│       │
│       ├── src/
│       │   └── ecommerce_service/
│       │       ├── __init__.py
│       │       ├── agents/
│       │       │   ├── __init__.py
│       │       │   └── customer_service_agent.py
│       │       └── tools/
│       │           ├── __init__.py
│       │           └── customer_db_tool.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── middleware/
│       │   │   ├── __init__.py
│       │   │   ├── auth_middleware.py
│       │   │   ├── cors_middleware.py
│       │   │   └── rate_limit_middleware.py
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── agent.py
│       │   │   ├── chat.py
│       │   │   ├── health.py
│       │   │   └── websocket.py
│       │   ├── schemas/
│       │   │   ├── __init__.py
│       │   │   ├── request.py
│       │   │   └── response.py
│       │   └── websocket/
│       │       ├── __init__.py
│       │       ├── connection_manager.py
│       │       └── message_handler.py
│       │
│       ├── config/
│       │   ├── api.yaml
│       │   ├── cache.yaml
│       │   ├── domain.yaml
│       │   ├── llm.yaml
│       │   ├── rag.yaml
│       │   ├── rate_limit.yaml
│       │   └── websocket.yaml
│       │
│       ├── knowledge/
│       │   └── docs/
│       │       └── shipping_policy.md
│       │
│       ├── tests/
│       │   ├── conftest.py
│       │   ├── integration/
│       │   │   ├── test_sse_streaming.py
│       │   │   └── test_websocket.py
│       │   └── unit/
│       │       └── test_rate_limiter.py
│       │
│       └── docker/
│           └── docker-compose.yml
│
├── config/
│   └── base/
│       ├── api.yaml
│       ├── llm.yaml
│       ├── rag.yaml
│       └── rate_limit.yaml
│
├── infrastructure/
│   ├── docker/
│   │   └── docker-compose.base.yml
│   │
│   ├── k8s/
│   │   └── services/
│   │       └── ecommerce/
│   │           ├── deployment.yaml
│   │           ├── hpa.yaml
│   │           ├── istio-virtualservice.yaml
│   │           ├── rollout.yaml
│   │           └── service.yaml
│   │
│   └── nginx/
│       ├── nginx.conf
│       └── sites-enabled/
│           └── ecommerce.conf
│
├── scripts/
│   ├── deployment/
│   │   └── zero_downtime_deploy.sh
│   │
│   ├── migrations/
│   │
│   ├── monitoring/
│   │   ├── test_sse.py
│   │   └── test_websocket.py
│   │
│   └── setup/
│       └── init_redis.py
│
└── docs/
    ├── architecture/
    │   ├── api_design.md
    │   └── websocket_architecture.md
    │
    └── guides/
        ├── rate_limiting_guide.md
        ├── sse_guide.md
        └── websocket_guide.md
```

---

## 3. 코어 컴포넌트 구현

### 3.1 Orchestrator 시스템

#### 3.1.1 Base Orchestrator

```python
# packages/agentic-ai-core/src/agentic_ai_core/orchestrator/base_orchestrator.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

@dataclass
class AgentMessage:
    """Agent 간 표준 메시지 포맷"""
    sender: str
    receiver: str
    content: Any
    message_type: str  # "request", "response", "notification"
    timestamp: datetime = None
    trace_id: str = None
    
    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now()
        self.trace_id = self.trace_id or str(uuid4())


@dataclass
class ExecutionPlan:
    """실행 계획"""
    task: str
    subtasks: List[Dict[str, Any]]
    pattern: str  # "supervisor", "hierarchy", "collaborative", "sequential"


class BaseOrchestrator:
    """모든 도메인에서 재사용 가능한 orchestration 엔진"""
    
    def __init__(self, config):
        self.config = config
        self.pattern = config.pattern  # "supervisor", "hierarchy", etc.
        self.agents: Dict[str, Any] = {}
        self.workflow = None
        self.message_bus = MessageBus()
        
    def register_agent(self, name: str, agent_class, role: str, capabilities: List[str]):
        """Agent 등록 (도메인별로 다른 agent 주입)"""
        self.agents[name] = {
            "instance": agent_class(self.config),
            "role": role,
            "capabilities": capabilities
        }
    
    def load_workflow(self, workflow_config: dict):
        """워크플로우 정의 로드 (YAML/JSON)"""
        self.workflow = WorkflowEngine(workflow_config)
    
    async def execute(self, task: str, context: dict) -> Any:
        """패턴에 따라 자동으로 agent 조율"""
        
        if self.pattern == "supervisor":
            return await self._execute_supervisor(task, context)
        elif self.pattern == "hierarchy":
            return await self._execute_hierarchy(task, context)
        elif self.pattern == "collaborative":
            return await self._execute_collaborative(task, context)
        elif self.pattern == "sequential":
            return await self._execute_sequential(task, context)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
    
    async def _execute_supervisor(self, task: str, context: dict) -> Any:
        """Supervisor 패턴: 감독자가 작업 분해 후 전문 agent에 위임"""
        # 1. Supervisor가 작업 분석
        supervisor = self.agents["supervisor"]["instance"]
        plan = await supervisor.create_plan(task)
        
        # 2. 각 하위 작업을 전문 agent에게 위임
        results = {}
        for subtask in plan.subtasks:
            agent_name = self._select_agent(subtask)
            agent = self.agents[agent_name]["instance"]
            results[subtask["id"]] = await agent.execute(subtask, context)
        
        # 3. Supervisor가 결과 통합
        final_result = await supervisor.synthesize(results)
        return final_result
    
    async def _execute_hierarchy(self, task: str, context: dict) -> Any:
        """Hierarchy 패턴: 계층적 위임 (Level 1 → Level 2 → Level 3)"""
        top_agent = self._get_top_level_agent()
        return await top_agent.execute_with_delegation(task, context, self.agents)
    
    async def _execute_collaborative(self, task: str, context: dict) -> Any:
        """Collaborative 패턴: 여러 agent가 동시에 협업"""
        import asyncio
        
        # 모든 관련 agent가 동시에 작업
        relevant_agents = self._select_relevant_agents(task)
        tasks = [
            agent.execute(task, context) 
            for agent in relevant_agents
        ]
        results = await asyncio.gather(*tasks)
        
        # 결과 병합
        return self._merge_results(results)
    
    async def _execute_sequential(self, task: str, context: dict) -> Any:
        """Sequential 패턴: 순차적 처리 (파이프라인)"""
        result = task
        for step in self.workflow.steps:
            agent = self.agents[step.agent]["instance"]
            result = await agent.execute(result, context)
            context["previous_result"] = result
        return result
    
    def _select_agent(self, subtask: dict) -> str:
        """작업에 맞는 agent 자동 선택"""
        task_type = subtask.get("type")
        for name, agent_info in self.agents.items():
            if task_type in agent_info["capabilities"]:
                return name
        raise ValueError(f"No agent found for task type: {task_type}")
    
    def _get_top_level_agent(self):
        """최상위 레벨 agent 반환"""
        for name, info in self.agents.items():
            if info.get("level") == 1:
                return info["instance"]
        raise ValueError("No top-level agent found")
    
    def _select_relevant_agents(self, task: str):
        """작업에 관련된 모든 agent 선택"""
        return [info["instance"] for info in self.agents.values()]
    
    def _merge_results(self, results: List[Any]) -> Any:
        """여러 agent의 결과 병합"""
        return {"merged_results": results}


class MessageBus:
    """Agent 간 메시지 전달 (Pub/Sub)"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
    
    def register(self, name: str, agent):
        self.agents[name] = agent
    
    async def send(self, message: AgentMessage) -> Any:
        """메시지 전송"""
        receiver = self.agents.get(message.receiver)
        if not receiver:
            raise ValueError(f"Agent not found: {message.receiver}")
        return await receiver.handle_message(message)
    
    async def broadcast(self, sender: str, content: Any):
        """모든 agent에게 브로드캐스트"""
        for name, agent in self.agents.items():
            if name != sender:
                message = AgentMessage(
                    sender=sender,
                    receiver=name,
                    content=content,
                    message_type="notification"
                )
                await agent.handle_message(message)


class WorkflowEngine:
    """워크플로우 실행 엔진"""
    
    def __init__(self, config: dict):
        self.name = config.get("name", "default_workflow")
        self.steps = self._parse_steps(config.get("steps", []))
    
    def _parse_steps(self, steps_config: List[dict]):
        return [WorkflowStep(**step) for step in steps_config]


@dataclass
class WorkflowStep:
    """워크플로우 스텝"""
    agent: str
    action: str
    condition: Optional[str] = None
    params: Optional[dict] = None
```

#### 3.1.2 Secure Orchestrator (RBAC 적용)

```python
# packages/agentic-ai-core/src/agentic_ai_core/orchestrator/secure_orchestrator.py

from typing import Dict, Any, List
from .base_orchestrator import BaseOrchestrator, ExecutionPlan
from ..security.rbac import RBACManager, PermissionDenied
from ..security.audit.logger import AuditLogger

@dataclass
class ExecutionStep:
    """실행 스텝"""
    id: str
    type: str
    description: str = ""
    agent: str = ""
    action: str = ""
    params: dict = None


class SecureOrchestrator(BaseOrchestrator):
    """RBAC 적용된 보안 Orchestrator"""
    
    def __init__(self, config):
        super().__init__(config)
        self.rbac = RBACManager(config.security)
        self.audit_logger = AuditLogger()
    
    async def execute(self, task: str, user: Any, context: dict = None) -> Any:
        """보안 검증 후 실행"""
        context = context or {}
        context["user"] = user
        
        # 1. 실행 계획 수립
        plan = await self._create_execution_plan(task)
        
        # 2. 각 agent/tool에 대해 권한 검증
        for step in plan.steps:
            await self.rbac.authorize_request(
                user=user,
                requested_agent=step.agent,
                requested_action=step.action,
                params=step.params
            )
        
        # 3. 감사 로그 기록
        await self.audit_logger.log(
            user_id=user.id,
            action="execute_task",
            resource=task,
            granted=True
        )
        
        # 4. 실행
        result = await super().execute(task, context)
        
        # 5. 결과 필터링 (역할에 따라 민감 정보 제거)
        filtered_result = await self._filter_response(result, user.role)
        
        return filtered_result
    
    async def _create_execution_plan(self, task: str) -> ExecutionPlan:
        """실행 계획 수립"""
        supervisor = self.agents.get("supervisor", {}).get("instance")
        if supervisor:
            return await supervisor.create_plan(task)
        
        return ExecutionPlan(
            task=task,
            subtasks=[{"id": "main", "type": "general", "description": task}],
            pattern=self.pattern
        )
    
    async def _filter_response(self, result: dict, role: str) -> dict:
        """역할에 따라 응답 필터링"""
        if not isinstance(result, dict):
            return result
            
        if role == "customer":
            return self._mask_sensitive_data(result, [
                "customer_email", "phone_number", "internal_notes"
            ])
        elif role == "merchant":
            return self._mask_sensitive_data(result, [
                "customer_ssn", "card_number"
            ])
        return result
    
    def _mask_sensitive_data(self, data: dict, fields: List[str]) -> dict:
        """민감 정보 마스킹"""
        masked = data.copy()
        for field in fields:
            if field in masked:
                value = masked[field]
                if isinstance(value, str):
                    if "email" in field:
                        masked[field] = self._mask_email(value)
                    elif "phone" in field:
                        masked[field] = self._mask_phone(value)
                    else:
                        masked[field] = "***"
        return masked
    
    def _mask_email(self, email: str) -> str:
        if "@" not in email:
            return "***"
        local, domain = email.split("@")
        return f"{local[:2]}***@{domain}"
    
    def _mask_phone(self, phone: str) -> str:
        if len(phone) < 8:
            return "***"
        return f"{phone[:3]}-****-{phone[-4:]}"
```

#### 3.1.3 워크플로우 설정 예시 (YAML)

```yaml
# config/workflows/ecommerce_supervisor.yaml

orchestration:
  pattern: "supervisor"
  
agents:
  supervisor:
    class: "ecommerce_service.agents.StoreManagerAgent"
    role: "감독자"
    capabilities: ["task_decomposition", "result_synthesis"]
    
  customer_service:
    class: "ecommerce_service.agents.CustomerServiceAgent"
    role: "고객 응대"
    capabilities:
      - "customer_inquiry"
      - "order_lookup"
      - "complaint_handling"
      - "refund_processing"
  
  inventory_manager:
    class: "ecommerce_service.agents.InventoryManagerAgent"
    role: "재고 관리"
    capabilities:
      - "stock_check"
      - "reorder_suggestion"
  
  marketing_analyst:
    class: "ecommerce_service.agents.MarketingAnalystAgent"
    role: "마케팅 분석"
    capabilities:
      - "campaign_analysis"
      - "customer_segmentation"

workflows:
  customer_inquiry:
    steps:
      - agent: "customer_service"
        action: "handle_inquiry"
      - agent: "supervisor"
        action: "review_response"
        condition: "if response.needs_review"
```

---

### 3.2 Agent 시스템

#### 3.2.1 Base Agent

```python
# packages/agentic-ai-core/src/agentic_ai_core/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..tools.base_tool import BaseTool
from ..llm.gateway import LLMGateway
from ..rag.retriever import Retriever
from ..observability.tracer import trace_agent

@dataclass
class AgentConfig:
    """Agent 설정"""
    name: str
    description: str
    capabilities: List[str]
    llm_config: dict
    rag_config: dict
    tools_config: dict


class BaseAgent(ABC):
    """모든 Agent의 기본 클래스"""
    
    # 서브클래스에서 오버라이드
    NAME: str = "base_agent"
    DESCRIPTION: str = "Base agent class"
    CAPABILITIES: List[str] = []
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = self.NAME
        self.description = self.DESCRIPTION
        self.capabilities = self.CAPABILITIES
        
        # 핵심 컴포넌트 초기화
        self.llm = LLMGateway(config.llm_config)
        self.rag = Retriever(config.rag_config)
        self.tools: Dict[str, BaseTool] = {}
        
        # 시스템 프롬프트
        self.system_prompt: str = ""
    
    def register_tools(self, tools: List[BaseTool]):
        """Tool 등록"""
        for tool in tools:
            self.tools[tool.name] = tool
    
    def get_tool_descriptions(self) -> str:
        """LLM이 이해할 수 있는 형식으로 Tool 정보 제공"""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"Tool: {name}\n"
            desc += f"Description: {tool.description}\n"
            desc += f"Functions: {tool.get_function_descriptions()}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    @trace_agent(name="agent_execution")
    async def execute(self, task: str, context: dict) -> str:
        """Agent 작업 실행"""
        
        # 1. RAG로 관련 문서 검색
        relevant_docs = await self.rag.retrieve(
            query=task,
            domain=context.get("domain", "default"),
            top_k=3
        )
        
        # 2. LLM으로 작업 계획 수립
        plan = await self._create_plan(task, relevant_docs)
        
        # 3. Tool 실행
        tool_results = await self._execute_tools(plan)
        
        # 4. 최종 응답 생성
        response = await self._generate_response(
            task=task,
            context=context,
            relevant_docs=relevant_docs,
            tool_results=tool_results
        )
        
        return response
    
    async def _create_plan(self, task: str, context_docs: List[dict]) -> dict:
        """작업 계획 수립"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Task: {task}

Available tools:
{self.get_tool_descriptions()}

Context:
{self._format_docs(context_docs)}

Create an execution plan. Return a JSON with:
- intent: what the user wants
- tools_to_use: list of tool calls with function and arguments
"""}
        ]
        
        response = await self.llm.generate(messages=messages, temperature=0.3)
        
        import json
        try:
            return json.loads(response.content)
        except:
            return {"intent": task, "tools_to_use": []}
    
    async def _execute_tools(self, plan: dict) -> Dict[str, Any]:
        """Tool 실행"""
        results = {}
        
        for tool_call in plan.get("tools_to_use", []):
            tool_name = tool_call.get("tool")
            function_name = tool_call.get("function")
            arguments = tool_call.get("arguments", {})
            
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    result = await getattr(tool, function_name)(**arguments)
                    results[f"{tool_name}.{function_name}"] = result
                except Exception as e:
                    results[f"{tool_name}.{function_name}"] = {"error": str(e)}
        
        return results
    
    async def _generate_response(
        self,
        task: str,
        context: dict,
        relevant_docs: List[dict],
        tool_results: Dict[str, Any]
    ) -> str:
        """최종 응답 생성"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}"},
            {"role": "assistant", "content": f"Tool results: {tool_results}"},
            {"role": "user", "content": f"""
Relevant context: {self._format_docs(relevant_docs)}

Based on the tool results and context, provide a helpful response.
"""}
        ]
        
        response = await self.llm.generate(messages=messages, temperature=0.7)
        return response.content
    
    def _format_docs(self, docs: List[dict]) -> str:
        """문서 포맷팅"""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", doc.get("content", ""))
            source = doc.get("metadata", {}).get("source", "unknown")
            formatted.append(f"[{i}] ({source}): {text[:500]}...")
        
        return "\n".join(formatted)
    
    async def create_plan(self, task: str) -> ExecutionPlan:
        """Supervisor용: 작업을 하위 작업으로 분해"""
        messages = [
            {"role": "system", "content": "You are a task planner. Break down tasks into subtasks."},
            {"role": "user", "content": task}
        ]
        
        response = await self.llm.generate(messages=messages, temperature=0.3)
        
        from ..orchestrator.base_orchestrator import ExecutionPlan
        return ExecutionPlan(
            task=task,
            subtasks=[{"id": "1", "type": "general", "description": task}],
            pattern="supervisor"
        )
    
    async def synthesize(self, results: Dict[str, Any]) -> str:
        """Supervisor용: 여러 결과를 통합"""
        messages = [
            {"role": "system", "content": "Combine results into a coherent response."},
            {"role": "user", "content": f"Results: {results}"}
        ]
        
        response = await self.llm.generate(messages=messages, temperature=0.7)
        return response.content
    
    async def handle_message(self, message) -> Any:
        """다른 Agent로부터 메시지 처리"""
        if message.message_type == "request":
            return await self.execute(message.content, {"sender": message.sender})
        return None


class AgentRegistry:
    """Agent 등록 및 관리 (싱글톤)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.agents = {}
        return cls._instance
    
    def register(self, agent_class: type):
        self.agents[agent_class.NAME] = agent_class
    
    def get(self, name: str) -> Optional[type]:
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        return list(self.agents.keys())
```

---

### 3.3 Tool 시스템

#### 3.3.1 Base Tool

```python
# packages/agentic-ai-core/src/agentic_ai_core/tools/base_tool.py

from abc import ABC
from typing import Any, Dict, List
import inspect

class BaseTool(ABC):
    """모든 Tool의 기본 클래스"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.description = self.__class__.__doc__ or "No description"
    
    def get_function_descriptions(self) -> str:
        """Tool이 제공하는 함수 목록"""
        functions = []
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '__doc__'):
                sig = inspect.signature(attr)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    param_str = param_name
                    if param.annotation != inspect.Parameter.empty:
                        param_str += f": {param.annotation.__name__}"
                    if param.default != inspect.Parameter.empty:
                        param_str += f" = {param.default}"
                    params.append(param_str)
                
                func_desc = f"  - {attr_name}({', '.join(params)})"
                if attr.__doc__:
                    func_desc += f": {attr.__doc__.strip().split(chr(10))[0]}"
                functions.append(func_desc)
        
        return "\n".join(functions)
    
    def get_openai_function_schema(self) -> List[dict]:
        """OpenAI Function Calling 스키마 생성"""
        functions = []
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '__doc__'):
                sig = inspect.signature(attr)
                
                parameters = {"type": "object", "properties": {}, "required": []}
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        type_map = {
                            str: "string", int: "integer", float: "number",
                            bool: "boolean", list: "array", dict: "object"
                        }
                        param_type = type_map.get(param.annotation, "string")
                    
                    parameters["properties"][param_name] = {"type": param_type}
                    
                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(param_name)
                
                functions.append({
                    "name": f"{self.name}_{attr_name}",
                    "description": attr.__doc__ or f"Execute {attr_name}",
                    "parameters": parameters
                })
        
        return functions
```

#### 3.3.2 Tool Registry

```python
# packages/agentic-ai-core/src/agentic_ai_core/tools/tool_registry.py

from typing import Dict, List, Optional
from .base_tool import BaseTool

class ToolRegistry:
    """Tool 등록 및 관리"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Tool 등록"""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Tool 조회"""
        return self.tools.get(name)
    
    def get_all_function_schemas(self) -> List[dict]:
        """모든 Tool의 Function Calling 스키마"""
        schemas = []
        for tool in self.tools.values():
            schemas.extend(tool.get_openai_function_schema())
        return schemas
    
    def get_descriptions_for_llm(self) -> str:
        """LLM용 Tool 설명"""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"Tool: {name}\n"
            desc += f"Description: {tool.description}\n"
            desc += f"Functions:\n{tool.get_function_descriptions()}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)
```

#### 3.3.3 Tool Security Wrapper

```python
# packages/agentic-ai-core/src/agentic_ai_core/tools/tool_security_wrapper.py

from typing import Any, Dict, List
from .base_tool import BaseTool
from ..security.audit.logger import AuditLogger

class ToolSecurityWrapper:
    """Tool 실행 시 자동으로 권한 체크 및 감사 로깅"""
    
    def __init__(self, tool: BaseTool, permission_config: Dict[str, List[str]]):
        self.tool = tool
        self.permissions = permission_config  # {function_name: [allowed_roles]}
        self.audit_logger = AuditLogger()
    
    async def execute(self, function_name: str, user: Any, **kwargs) -> Any:
        """권한 검증 후 Tool 실행"""
        
        # 1. 권한 확인
        if not self._has_permission(user, function_name):
            await self.audit_logger.log(
                user_id=user.id,
                action=f"tool_{self.tool.name}_{function_name}",
                resource=str(kwargs),
                granted=False,
                reason="Permission denied"
            )
            raise PermissionDenied(
                f"User {user.id} cannot execute {self.tool.name}.{function_name}"
            )
        
        # 2. 실제 실행
        func = getattr(self.tool, function_name)
        result = await func(**kwargs)
        
        # 3. 감사 로그
        await self.audit_logger.log(
            user_id=user.id,
            action=f"tool_{self.tool.name}_{function_name}",
            resource=str(kwargs),
            granted=True,
            result_summary=str(result)[:200]
        )
        
        return result
    
    def _has_permission(self, user: Any, function_name: str) -> bool:
        allowed_roles = self.permissions.get(function_name, [])
        if not allowed_roles or "all" in allowed_roles:
            return True
        return user.role in allowed_roles


class PermissionDenied(Exception):
    pass
```

---

## 4. RAG 시스템

### 4.1 Embedder

```python
# packages/agentic-ai-core/src/agentic_ai_core/rag/embedder.py

from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """임베딩 생성기 기본 클래스"""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class EmbedderFactory:
    """Embedder 팩토리"""
    
    @staticmethod
    def create(config: dict) -> BaseEmbedder:
        provider = config.get("provider", "openai")
        
        if provider == "openai":
            return OpenAIEmbedder(config)
        elif provider == "local":
            return LocalEmbedder(config)
        else:
            raise ValueError(f"Unknown provider: {provider}")


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 임베딩"""
    
    def __init__(self, config: dict):
        self.model = config.get("model", "text-embedding-3-small")
        self.api_key = config.get("api_key")
        
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def embed_text(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model=self.model, input=text
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.model, input=texts
        )
        return [item.embedding for item in response.data]


class LocalEmbedder(BaseEmbedder):
    """로컬 임베딩 (Sentence Transformers)"""
    
    def __init__(self, config: dict):
        self.model_name = config.get("model", "BAAI/bge-m3")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
    
    async def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
```

### 4.2 Milvus Store

```python
# packages/agentic-ai-core/src/agentic_ai_core/rag/milvus_store.py

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    """검색 결과"""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class MilvusStore:
    """Milvus 벡터 저장소"""
    
    def __init__(self, config: dict):
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.collection_name = config.get("collection_name", "documents")
        self.dimension = config.get("dimension", 1536)
        
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=config.get("user"),
            password=config.get("password")
        )
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """컬렉션 생성 또는 로드"""
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
        
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            schema = CollectionSchema(fields, description="Document embeddings")
            self.collection = Collection(self.collection_name, schema)
            
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
        
        self.collection.load()
    
    async def insert(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """문서 삽입"""
        domains = [m.get("domain", "default") for m in metadatas]
        sources = [m.get("source", "unknown") for m in metadatas]
        
        data = [ids, texts, domains, sources, embeddings]
        self.collection.insert(data)
        self.collection.flush()
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """벡터 검색"""
        expr = None
        if filters and "domain" in filters:
            expr = f'domain == "{filters["domain"]}"'
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "text", "domain", "source"]
        )
        
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    id=hit.entity.get("id"),
                    text=hit.entity.get("text"),
                    metadata={
                        "domain": hit.entity.get("domain"),
                        "source": hit.entity.get("source")
                    },
                    score=hit.distance
                ))
        
        return search_results
```

### 4.3 Document Chunker

```python
# packages/agentic-ai-core/src/agentic_ai_core/rag/chunker.py

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Chunk:
    """문서 청크"""
    text: str
    metadata: Dict[str, Any]
    embedding: List[float] = None


class DocumentChunker:
    """문서 분할기"""
    
    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "recursive")
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
    
    def chunk_document(self, document: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """문서 분할"""
        if self.strategy == "fixed":
            return self._fixed_size_chunking(document, metadata)
        elif self.strategy == "recursive":
            return self._recursive_chunking(document, metadata)
        else:
            return self._recursive_chunking(document, metadata)
    
    def _fixed_size_chunking(self, document: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """고정 크기 분할"""
        chunks = []
        words = document.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks), "chunk_strategy": "fixed"}
            ))
        
        return chunks
    
    def _recursive_chunking(self, document: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """재귀적 분할"""
        separators = ["\n\n", "\n", ". ", " "]
        chunks = []
        self._split_recursive(document, separators, chunks, metadata)
        return chunks
    
    def _split_recursive(
        self, text: str, separators: List[str], 
        chunks: List[Chunk], metadata: Dict[str, Any]
    ):
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(Chunk(
                    text=text.strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "chunk_strategy": "recursive"}
                ))
            return
        
        separator = separators[0] if separators else " "
        parts = text.split(separator)
        
        current_chunk = []
        current_length = 0
        
        for part in parts:
            part_length = len(part)
            
            if current_length + part_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if len(chunk_text) > self.chunk_size and len(separators) > 1:
                        self._split_recursive(chunk_text, separators[1:], chunks, metadata)
                    else:
                        chunks.append(Chunk(
                            text=chunk_text.strip(),
                            metadata={**metadata, "chunk_index": len(chunks), "chunk_strategy": "recursive"}
                        ))
                current_chunk = [part]
                current_length = part_length
            else:
                current_chunk.append(part)
                current_length += part_length + len(separator)
        
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text.strip(),
                metadata={**metadata, "chunk_index": len(chunks), "chunk_strategy": "recursive"}
            ))
```

### 4.4 Retriever

```python
# packages/agentic-ai-core/src/agentic_ai_core/rag/retriever.py

from typing import List, Dict, Any
from dataclasses import dataclass
from .embedder import EmbedderFactory
from .milvus_store import MilvusStore, SearchResult

@dataclass
class Document:
    """검색된 문서"""
    text: str
    metadata: Dict[str, Any]
    score: float


class Retriever:
    """지능형 검색기"""
    
    def __init__(self, config: dict):
        self.config = config
        self.embedder = EmbedderFactory.create(config.get("embedding", {}))
        self.vector_store = MilvusStore(config.get("vector_store", {}))
        self.query_rewriter = QueryRewriter(config.get("query_rewriting", {}))
    
    async def retrieve(
        self,
        query: str,
        domain: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Document]:
        """검색 파이프라인"""
        
        # 1. 쿼리 재작성
        queries = await self.query_rewriter.rewrite(query)
        
        # 2. 각 쿼리로 검색
        all_results = []
        for q in queries:
            query_embedding = await self.embedder.embed_text(q)
            search_filters = {**(filters or {}), "domain": domain}
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                filters=search_filters
            )
            all_results.extend(results)
        
        # 3. 중복 제거
        unique_results = self._deduplicate(all_results)
        
        # 4. 상위 K개 반환
        documents = [
            Document(text=r.text, metadata=r.metadata, score=r.score)
            for r in unique_results[:top_k]
        ]
        
        return documents
    
    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        seen_ids = set()
        unique = []
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique.append(result)
        return unique


class QueryRewriter:
    """쿼리 최적화"""
    
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.methods = config.get("methods", ["expansion"])
    
    async def rewrite(self, query: str) -> List[str]:
        if not self.enabled:
            return [query]
        
        queries = [query]
        # 확장된 쿼리 추가 가능
        return queries
```

### 4.5 Document Indexer

```python
# packages/agentic-ai-core/src/agentic_ai_core/rag/indexer.py

from typing import List, Dict, Any
from pathlib import Path
from uuid import uuid4
from .embedder import EmbedderFactory
from .chunker import DocumentChunker
from .milvus_store import MilvusStore

class DocumentIndexer:
    """문서 인덱싱 파이프라인"""
    
    def __init__(self, config: dict):
        self.config = config
        self.embedder = EmbedderFactory.create(config.get("embedding", {}))
        self.chunker = DocumentChunker(config.get("chunking", {}))
        self.vector_store = MilvusStore(config.get("vector_store", {}))
    
    async def index_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        domain: str
    ) -> Dict[str, Any]:
        """문서 일괄 인덱싱"""
        all_chunks = []
        
        # 1. 문서 분할
        for doc, metadata in zip(documents, metadatas):
            metadata["domain"] = domain
            chunks = self.chunker.chunk_document(doc, metadata)
            all_chunks.extend(chunks)
        
        # 2. 임베딩 생성 (배치)
        texts = [chunk.text for chunk in all_chunks]
        embeddings = await self.embedder.embed_batch(texts)
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # 3. 벡터 DB에 저장
        ids = [str(uuid4()) for _ in all_chunks]
        chunk_texts = [c.text for c in all_chunks]
        chunk_embeddings = [c.embedding for c in all_chunks]
        chunk_metadatas = [c.metadata for c in all_chunks]
        
        await self.vector_store.insert(
            ids=ids,
            texts=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=chunk_metadatas
        )
        
        return {
            "indexed_documents": len(documents),
            "total_chunks": len(all_chunks),
            "domain": domain
        }
    
    async def index_from_directory(self, path: str, domain: str) -> Dict[str, Any]:
        """디렉토리에서 문서 인덱싱"""
        documents = []
        metadatas = []
        
        for file_path in Path(path).rglob("*"):
            if file_path.suffix in [".txt", ".md"]:
                content = file_path.read_text(encoding="utf-8")
                documents.append(content)
                metadatas.append({
                    "source": str(file_path),
                    "file_type": file_path.suffix
                })
        
        return await self.index_documents(documents, metadatas, domain)
```

---

## 5. LLM Gateway

### 5.1 Gateway

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/gateway.py

from typing import List, Dict, Any
from dataclasses import dataclass
import time
import hashlib
import json
from .router import LLMRouter
from .cache import LLMCache

@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    provider: str
    tokens_used: Dict[str, int]
    cost: float
    latency_ms: float
    from_cache: bool = False
    attempt: int = 1


class LLMGateway:
    """모든 LLM을 통합 관리하는 게이트웨이"""
    
    def __init__(self, config: dict):
        self.config = config
        self.primary_provider = config.get("primary_provider", "openai")
        self.fallback_providers = config.get("fallback_providers", [])
        self.providers = {}
        self.router = LLMRouter(config.get("routing", {}))
        self.cache = LLMCache(config.get("cache", {}))
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """프로바이더 초기화"""
        from .providers.openai_provider import OpenAIProvider
        from .providers.anthropic_provider import AnthropicProvider
        from .providers.vllm_provider import VLLMProvider
        from .providers.ollama_provider import OllamaProvider
        
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "local_vllm": VLLMProvider,
            "ollama": OllamaProvider
        }
        
        for provider_config in self.config.get("providers", []):
            provider_type = provider_config.get("type")
            if provider_type in provider_classes and provider_config.get("enabled", True):
                self.providers[provider_type] = provider_classes[provider_type](provider_config)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """LLM 호출 (자동 라우팅, 폴백, 캐싱)"""
        
        # 1. 캐시 확인
        cache_key = self._make_cache_key(messages, model, temperature, max_tokens)
        cached = await self.cache.get(cache_key)
        if cached:
            return LLMResponse(
                content=cached["content"],
                model=cached["model"],
                provider=cached["provider"],
                tokens_used=cached["tokens_used"],
                cost=cached["cost"],
                latency_ms=0,
                from_cache=True
            )
        
        # 2. 라우팅
        provider_name = self.router.select_provider(model=model, context=kwargs)
        
        # 3. 폴백을 포함한 호출
        response = await self._generate_with_fallback(
            provider_name=provider_name,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 4. 캐싱
        await self.cache.set(cache_key, {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "tokens_used": response.tokens_used,
            "cost": response.cost
        })
        
        return response
    
    async def _generate_with_fallback(
        self, provider_name: str, messages: List[Dict],
        model: str, temperature: float, max_tokens: int, **kwargs
    ) -> LLMResponse:
        """폴백을 포함한 호출"""
        
        providers_to_try = [provider_name] + self.fallback_providers
        
        for attempt, prov_name in enumerate(providers_to_try, 1):
            if prov_name not in self.providers:
                continue
            
            provider = self.providers[prov_name]
            
            try:
                response = await provider.generate(
                    messages=messages,
                    model=model or provider.config.get("default_model"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                return LLMResponse(
                    content=response.content,
                    model=response.model,
                    provider=prov_name,
                    tokens_used=response.tokens_used,
                    cost=response.cost,
                    latency_ms=response.latency_ms,
                    attempt=attempt
                )
            
            except Exception as e:
                print(f"Provider {prov_name} failed: {e}")
                if attempt == len(providers_to_try):
                    raise LLMGatewayError(f"All providers failed. Last error: {e}")
    
    def _make_cache_key(self, messages, model, temperature, max_tokens) -> str:
        data = {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class LLMGatewayError(Exception):
    pass
```

### 5.2 LLM Router

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/router.py

from typing import Dict, Any

class LLMRouter:
    """요청에 따라 최적의 LLM 프로바이더 선택"""
    
    def __init__(self, config: dict):
        self.config = config
        self.strategy = config.get("strategy", "cost")
        self.default_provider = config.get("default_provider", "openai")
    
    def select_provider(self, model: str = None, context: dict = None) -> str:
        """최적의 프로바이더 선택"""
        context = context or {}
        
        # 명시적으로 지정된 경우
        if "provider" in context:
            return context["provider"]
        
        if self.strategy == "cost":
            return self._select_by_cost(context)
        elif self.strategy == "performance":
            return self._select_by_performance(context)
        elif self.strategy == "hybrid":
            return self._select_hybrid(context)
        
        return self.default_provider
    
    def _select_by_cost(self, context: dict) -> str:
        """비용 기준 선택: 로컬 LLM 우선"""
        if self._is_available("local_vllm"):
            return "local_vllm"
        if self._is_available("ollama"):
            return "ollama"
        return self.default_provider
    
    def _select_by_performance(self, context: dict) -> str:
        """성능 기준 선택"""
        complexity = context.get("complexity", "medium")
        
        if complexity == "high":
            return "anthropic" if self._is_available("anthropic") else "openai"
        elif complexity == "low":
            return "local_vllm" if self._is_available("local_vllm") else self.default_provider
        
        return self.default_provider
    
    def _select_hybrid(self, context: dict) -> str:
        """비용과 성능 균형"""
        budget = context.get("max_cost", float('inf'))
        
        if budget < 0.01 and self._is_available("local_vllm"):
            return "local_vllm"
        
        return self.default_provider
    
    def _is_available(self, provider: str) -> bool:
        return True  # 실제로는 헬스체크
```

### 5.3 LLM Providers

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/providers/base.py

from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ProviderResponse:
    content: str
    model: str
    tokens_used: Dict[str, int]
    cost: float
    latency_ms: float


class BaseLLMProvider(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def generate(self, messages: List[Dict], model: str, 
                       temperature: float, max_tokens: int, **kwargs) -> ProviderResponse:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
```

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/providers/openai_provider.py

import time
from typing import List, Dict
from .base import BaseLLMProvider, ProviderResponse

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API 프로바이더"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=config.get("api_key"))
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
    
    async def generate(
        self, messages: List[Dict], model: str = "gpt-4o-mini",
        temperature: float = 0.7, max_tokens: int = 1000, **kwargs
    ) -> ProviderResponse:
        start_time = time.time()
        
        response = await self.client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        tokens_used = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
        
        cost = self._calculate_cost(tokens_used, model)
        
        return ProviderResponse(
            content=response.choices[0].message.content,
            model=model, tokens_used=tokens_used,
            cost=cost, latency_ms=latency_ms
        )
    
    def _calculate_cost(self, tokens_used: Dict, model: str) -> float:
        pricing = self.pricing.get(model, self.pricing["gpt-4o-mini"])
        input_cost = (tokens_used["input"] / 1_000_000) * pricing["input"]
        output_cost = (tokens_used["output"] / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except:
            return False
```

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/providers/anthropic_provider.py

import time
from typing import List, Dict
from .base import BaseLLMProvider, ProviderResponse

class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) API 프로바이더"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=config.get("api_key"))
        self.pricing = {
            "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
            "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        }
    
    async def generate(
        self, messages: List[Dict], model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7, max_tokens: int = 1000, **kwargs
    ) -> ProviderResponse:
        start_time = time.time()
        
        # system 메시지 분리
        system_message = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        response = await self.client.messages.create(
            model=model, system=system_message, messages=user_messages,
            temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        tokens_used = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens
        }
        
        cost = self._calculate_cost(tokens_used, model)
        
        return ProviderResponse(
            content=response.content[0].text,
            model=model, tokens_used=tokens_used,
            cost=cost, latency_ms=latency_ms
        )
    
    def _calculate_cost(self, tokens_used: Dict, model: str) -> float:
        pricing = self.pricing.get(model, self.pricing["claude-sonnet-4-5-20250929"])
        input_cost = (tokens_used["input"] / 1_000_000) * pricing["input"]
        output_cost = (tokens_used["output"] / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    async def health_check(self) -> bool:
        try:
            # 간단한 테스트 호출
            return True
        except:
            return False
```

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/providers/ollama_provider.py

import time
from typing import List, Dict
import httpx
from .base import BaseLLMProvider, ProviderResponse

class OllamaProvider(BaseLLMProvider):
    """로컬 Ollama 프로바이더"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "llama3:70b")
        self.client = httpx.AsyncClient()
    
    async def generate(
        self, messages: List[Dict], model: str = None,
        temperature: float = 0.7, max_tokens: int = 1000, **kwargs
    ) -> ProviderResponse:
        start_time = time.time()
        
        prompt = self._messages_to_prompt(messages)
        
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model or self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            }
        )
        
        result = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        return ProviderResponse(
            content=result["response"],
            model=self.model_name,
            tokens_used={
                "input": result.get("prompt_eval_count", 0),
                "output": result.get("eval_count", 0),
                "total": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            },
            cost=0.0,  # 로컬이라 무료
            latency_ms=latency_ms
        )
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        parts = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
```

### 5.4 LLM Cache

```python
# packages/agentic-ai-core/src/agentic_ai_core/llm/cache.py

from typing import Optional, Dict, Any
import json

class LLMCache:
    """LLM 응답 캐싱"""
    
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.ttl = config.get("ttl", 3600)
        self.use_redis = config.get("use_redis", False)
        
        if self.use_redis:
            import redis.asyncio as redis
            self.redis = redis.from_url(config.get("redis_url", "redis://localhost:6379"))
        else:
            self._local_cache = {}
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        
        if self.use_redis:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        else:
            return self._local_cache.get(key)
    
    async def set(self, key: str, value: Dict[str, Any]):
        if not self.enabled:
            return
        
        if self.use_redis:
            await self.redis.setex(key, self.ttl, json.dumps(value))
        else:
            self._local_cache[key] = value
```
## 6. 보안 시스템

### 6.1 RBAC (Role-Based Access Control)

```python
# packages/agentic-ai-core/src/agentic_ai_core/security/rbac.py

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class User:
    """사용자 정보"""
    id: int
    role: str
    email: str
    permissions: List[str] = None


@dataclass
class Permission:
    """권한 정의"""
    resource: str
    action: str
    conditions: Optional[Dict[str, Any]] = None


class RBACManager:
    """역할 기반 접근 제어"""
    
    def __init__(self, config: dict):
        self.config = config
        self.roles = self._load_roles(config.get("roles_config"))
        self.permissions = self._load_permissions(config.get("permissions_config"))
    
    def _load_roles(self, config_path: str) -> Dict[str, Dict]:
        """역할 설정 로드"""
        # YAML 또는 JSON에서 로드
        # 기본값 반환
        return {
            "customer": {
                "description": "일반 고객",
                "allowed_agents": ["customer_service"],
                "allowed_tools": ["order_lookup", "product_search"],
                "data_scope": "own"  # 자신의 데이터만
            },
            "merchant": {
                "description": "쇼핑몰 운영자",
                "allowed_agents": ["customer_service", "inventory_manager", "marketing_analyst"],
                "allowed_tools": ["order_lookup", "customer_db", "inventory_checker", "coupon_generator"],
                "data_scope": "all"  # 모든 데이터
            },
            "admin": {
                "description": "시스템 관리자",
                "allowed_agents": ["*"],  # 모든 agent
                "allowed_tools": ["*"],   # 모든 tool
                "data_scope": "all"
            }
        }
    
    def _load_permissions(self, config_path: str) -> Dict[str, List[Permission]]:
        """권한 설정 로드"""
        return {}
    
    async def authorize_request(
        self,
        user: User,
        requested_agent: str,
        requested_action: str,
        params: dict = None
    ) -> bool:
        """요청 권한 검증"""
        role_config = self.roles.get(user.role)
        
        if not role_config:
            raise PermissionDenied(f"Unknown role: {user.role}")
        
        # 1. Agent 접근 권한 확인
        allowed_agents = role_config.get("allowed_agents", [])
        if "*" not in allowed_agents and requested_agent not in allowed_agents:
            raise PermissionDenied(
                f"Role '{user.role}' cannot access agent '{requested_agent}'"
            )
        
        # 2. 데이터 스코프 확인
        data_scope = role_config.get("data_scope", "own")
        if data_scope == "own" and params:
            # 자신의 데이터에만 접근 가능
            target_user_id = params.get("user_id") or params.get("customer_id")
            if target_user_id and target_user_id != user.id:
                raise PermissionDenied("You can only access your own data")
        
        return True
    
    def check_tool_permission(
        self,
        user: User,
        tool_name: str,
        function_name: str
    ) -> bool:
        """Tool 사용 권한 확인"""
        role_config = self.roles.get(user.role)
        
        if not role_config:
            return False
        
        allowed_tools = role_config.get("allowed_tools", [])
        return "*" in allowed_tools or tool_name in allowed_tools
    
    def get_data_scope(self, user: User) -> str:
        """사용자의 데이터 접근 범위"""
        role_config = self.roles.get(user.role, {})
        return role_config.get("data_scope", "own")


class PermissionDenied(Exception):
    """권한 거부 예외"""
    pass
```

### 6.2 PBAC (Policy-Based Access Control)

```python
# packages/agentic-ai-core/src/agentic_ai_core/security/pbac_engine.py

from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Policy:
    """정책 정의"""
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    effect: str  # "allow" or "deny"
    priority: int = 0


class PBACEngine:
    """정책 기반 접근 제어 엔진"""
    
    def __init__(self, config: dict):
        self.policies = self._load_policies(config.get("policies", []))
        self.condition_evaluators = self._setup_evaluators()
    
    def _load_policies(self, policies_config: List[dict]) -> List[Policy]:
        """정책 로드"""
        policies = []
        
        # 기본 정책 예시
        default_policies = [
            Policy(
                name="working_hours_only",
                description="근무 시간에만 민감 데이터 접근 허용",
                conditions=[
                    {"type": "time_range", "start": "09:00", "end": "18:00"},
                    {"type": "weekday", "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]}
                ],
                effect="allow",
                priority=10
            ),
            Policy(
                name="refund_limit",
                description="환불 금액 제한",
                conditions=[
                    {"type": "amount_limit", "max_amount": 100000}
                ],
                effect="allow",
                priority=5
            ),
            Policy(
                name="vip_only_features",
                description="VIP 고객만 접근 가능한 기능",
                conditions=[
                    {"type": "user_tier", "required_tier": "VIP"}
                ],
                effect="allow",
                priority=15
            )
        ]
        
        return default_policies + [Policy(**p) for p in policies_config]
    
    def _setup_evaluators(self) -> Dict[str, Callable]:
        """조건 평가기 설정"""
        return {
            "time_range": self._evaluate_time_range,
            "weekday": self._evaluate_weekday,
            "amount_limit": self._evaluate_amount_limit,
            "user_tier": self._evaluate_user_tier,
            "ip_whitelist": self._evaluate_ip_whitelist
        }
    
    def evaluate(
        self,
        user: Any,
        action: str,
        resource: str,
        context: Dict[str, Any]
    ) -> bool:
        """정책 평가"""
        
        # 관련 정책 필터링
        applicable_policies = [
            p for p in self.policies
            if self._is_applicable(p, action, resource)
        ]
        
        # 우선순위 순으로 정렬
        applicable_policies.sort(key=lambda x: -x.priority)
        
        # 정책 평가
        for policy in applicable_policies:
            all_conditions_met = all(
                self._evaluate_condition(condition, user, context)
                for condition in policy.conditions
            )
            
            if all_conditions_met:
                return policy.effect == "allow"
        
        # 기본: 허용
        return True
    
    def _is_applicable(self, policy: Policy, action: str, resource: str) -> bool:
        """정책 적용 여부"""
        # 간단히 모든 정책 적용 (실제로는 더 정교하게)
        return True
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        user: Any,
        context: Dict[str, Any]
    ) -> bool:
        """조건 평가"""
        condition_type = condition.get("type")
        evaluator = self.condition_evaluators.get(condition_type)
        
        if not evaluator:
            return True
        
        return evaluator(condition, user, context)
    
    def _evaluate_time_range(
        self,
        condition: Dict,
        user: Any,
        context: Dict
    ) -> bool:
        """시간 범위 조건"""
        now = datetime.now().time()
        start = datetime.strptime(condition["start"], "%H:%M").time()
        end = datetime.strptime(condition["end"], "%H:%M").time()
        
        return start <= now <= end
    
    def _evaluate_weekday(
        self,
        condition: Dict,
        user: Any,
        context: Dict
    ) -> bool:
        """요일 조건"""
        today = datetime.now().strftime("%A").lower()
        allowed_days = [d.lower() for d in condition.get("days", [])]
        
        return today in allowed_days
    
    def _evaluate_amount_limit(
        self,
        condition: Dict,
        user: Any,
        context: Dict
    ) -> bool:
        """금액 제한 조건"""
        amount = context.get("amount", 0)
        max_amount = condition.get("max_amount", float('inf'))
        
        return amount <= max_amount
    
    def _evaluate_user_tier(
        self,
        condition: Dict,
        user: Any,
        context: Dict
    ) -> bool:
        """사용자 등급 조건"""
        required_tier = condition.get("required_tier")
        user_tier = getattr(user, "tier", None) or context.get("user_tier")
        
        tier_hierarchy = {"regular": 0, "silver": 1, "gold": 2, "VIP": 3}
        
        required_level = tier_hierarchy.get(required_tier, 0)
        user_level = tier_hierarchy.get(user_tier, 0)
        
        return user_level >= required_level
    
    def _evaluate_ip_whitelist(
        self,
        condition: Dict,
        user: Any,
        context: Dict
    ) -> bool:
        """IP 화이트리스트 조건"""
        client_ip = context.get("client_ip")
        allowed_ips = condition.get("allowed_ips", [])
        
        return client_ip in allowed_ips
```

### 6.3 Hybrid Access Control

```python
# packages/agentic-ai-core/src/agentic_ai_core/security/hybrid_access_control.py

from typing import Any, Dict
from .rbac import RBACManager, User, PermissionDenied
from .pbac_engine import PBACEngine

class HybridAccessControl:
    """RBAC + PBAC 결합"""
    
    def __init__(self, config: dict):
        self.rbac = RBACManager(config.get("rbac", {}))
        self.pbac = PBACEngine(config.get("pbac", {}))
    
    async def authorize(
        self,
        user: User,
        action: str,
        resource: str,
        context: Dict[str, Any] = None
    ) -> bool:
        """통합 권한 검증"""
        context = context or {}
        
        # 1단계: RBAC 검증 (빠른 검증)
        try:
            await self.rbac.authorize_request(
                user=user,
                requested_agent=context.get("agent", ""),
                requested_action=action,
                params=context
            )
        except PermissionDenied:
            return False
        
        # 2단계: PBAC 검증 (세밀한 정책)
        if not self.pbac.evaluate(user, action, resource, context):
            return False
        
        return True
    
    async def authorize_with_reason(
        self,
        user: User,
        action: str,
        resource: str,
        context: Dict[str, Any] = None
    ) -> tuple[bool, str]:
        """이유와 함께 권한 검증"""
        context = context or {}
        
        # RBAC 검증
        try:
            await self.rbac.authorize_request(
                user=user,
                requested_agent=context.get("agent", ""),
                requested_action=action,
                params=context
            )
        except PermissionDenied as e:
            return False, f"RBAC: {str(e)}"
        
        # PBAC 검증
        if not self.pbac.evaluate(user, action, resource, context):
            return False, "PBAC: Policy conditions not met"
        
        return True, "Authorized"
```

### 6.4 Audit Logger

```python
# packages/agentic-ai-core/src/agentic_ai_core/security/audit/logger.py

from typing import Any, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json

@dataclass
class AuditLog:
    """감사 로그 엔트리"""
    timestamp: str
    user_id: int
    user_role: str = ""
    action: str = ""
    resource: str = ""
    granted: bool = True
    reason: str = ""
    client_ip: str = ""
    result_summary: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """감사 로깅"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.storage = self._setup_storage()
    
    def _setup_storage(self):
        """저장소 설정"""
        storage_type = self.config.get("storage", "file")
        
        if storage_type == "file":
            return FileAuditStorage(self.config.get("file_path", "audit.log"))
        elif storage_type == "database":
            return DatabaseAuditStorage(self.config.get("db_config"))
        else:
            return MemoryAuditStorage()
    
    async def log(
        self,
        user_id: int,
        action: str,
        resource: str,
        granted: bool,
        reason: str = "",
        client_ip: str = "",
        result_summary: str = "",
        **kwargs
    ):
        """감사 로그 기록"""
        log_entry = AuditLog(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            action=action,
            resource=resource,
            granted=granted,
            reason=reason,
            client_ip=client_ip,
            result_summary=result_summary,
            metadata=kwargs
        )
        
        await self.storage.save(log_entry)
    
    async def query(
        self,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ):
        """감사 로그 조회"""
        return await self.storage.query(
            user_id=user_id,
            action=action,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )


class FileAuditStorage:
    """파일 기반 저장소"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    async def save(self, log_entry: AuditLog):
        with open(self.file_path, "a") as f:
            f.write(log_entry.to_json() + "\n")
    
    async def query(self, **kwargs):
        logs = []
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    logs.append(json.loads(line))
        except FileNotFoundError:
            pass
        return logs


class MemoryAuditStorage:
    """메모리 기반 저장소 (테스트용)"""
    
    def __init__(self):
        self.logs = []
    
    async def save(self, log_entry: AuditLog):
        self.logs.append(log_entry.to_dict())
    
    async def query(self, **kwargs):
        return self.logs


class DatabaseAuditStorage:
    """데이터베이스 저장소"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
    
    async def save(self, log_entry: AuditLog):
        # 실제 DB 저장 구현
        pass
    
    async def query(self, **kwargs):
        # 실제 DB 조회 구현
        return []
```

---

## 7. Observability

### 7.1 Langfuse Integration

```python
# packages/agentic-ai-core/src/agentic_ai_core/observability/langfuse_client.py

from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import os

class LangfuseClient:
    """Langfuse 통합 클라이언트"""
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        self.enabled = config.get("enabled", True)
        
        if self.enabled:
            from langfuse import Langfuse
            
            self.client = Langfuse(
                public_key=config.get("public_key") or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=config.get("secret_key") or os.getenv("LANGFUSE_SECRET_KEY"),
                host=config.get("host") or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
        else:
            self.client = None
    
    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """새 트레이스 시작"""
        if not self.enabled:
            return DummyTrace()
        
        return self.client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags
        )
    
    def span(
        self,
        trace,
        name: str,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """스팬 생성"""
        if not self.enabled:
            return DummySpan()
        
        return trace.span(
            name=name,
            input=input,
            metadata=metadata
        )
    
    def generation(
        self,
        trace,
        name: str,
        model: str,
        input: Any,
        output: Any = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """LLM 생성 기록"""
        if not self.enabled:
            return DummyGeneration()
        
        return trace.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage,
            metadata=metadata
        )
    
    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """점수 기록"""
        if not self.enabled:
            return
        
        self.client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment
        )
    
    def flush(self):
        """버퍼 플러시"""
        if self.client:
            self.client.flush()


class DummyTrace:
    """Langfuse 비활성화시 더미 트레이스"""
    def span(self, **kwargs): return DummySpan()
    def generation(self, **kwargs): return DummyGeneration()
    def end(self): pass


class DummySpan:
    """더미 스팬"""
    def end(self, **kwargs): pass


class DummyGeneration:
    """더미 생성"""
    def end(self, **kwargs): pass
```

### 7.2 Tracer Decorator

```python
# packages/agentic-ai-core/src/agentic_ai_core/observability/tracer.py

from functools import wraps
from typing import Callable, Any
import time
from .langfuse_client import LangfuseClient

# 글로벌 클라이언트
_langfuse_client = None

def init_tracer(config: dict):
    """트레이서 초기화"""
    global _langfuse_client
    _langfuse_client = LangfuseClient(config)


def get_langfuse() -> LangfuseClient:
    """Langfuse 클라이언트 반환"""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = LangfuseClient()
    return _langfuse_client


def trace_agent(name: str):
    """Agent 실행 트레이싱 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = get_langfuse()
            
            # 컨텍스트에서 정보 추출
            context = kwargs.get("context", {})
            user_id = context.get("user_id")
            session_id = context.get("session_id")
            
            trace = langfuse.trace(
                name=name,
                user_id=str(user_id) if user_id else None,
                session_id=session_id,
                tags=["agent", name]
            )
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                trace.span(
                    trace=trace,
                    name="result",
                    input=kwargs.get("task"),
                    metadata={
                        "duration_ms": (time.time() - start_time) * 1000,
                        "success": True
                    }
                )
                
                return result
            
            except Exception as e:
                trace.span(
                    trace=trace,
                    name="error",
                    input=str(e),
                    metadata={"success": False}
                )
                raise
            
            finally:
                trace.end()
        
        return wrapper
    return decorator


def trace_llm(name: str = "llm_call"):
    """LLM 호출 트레이싱 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = get_langfuse()
            
            trace = langfuse.trace(name=name, tags=["llm"])
            
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")
            
            try:
                result = await func(*args, **kwargs)
                
                langfuse.generation(
                    trace=trace,
                    name="generation",
                    model=model,
                    input=messages,
                    output=result.content if hasattr(result, "content") else str(result),
                    usage=result.tokens_used if hasattr(result, "tokens_used") else None,
                    metadata={
                        "cost": result.cost if hasattr(result, "cost") else 0,
                        "latency_ms": result.latency_ms if hasattr(result, "latency_ms") else 0
                    }
                )
                
                return result
            
            finally:
                trace.end()
        
        return wrapper
    return decorator


def trace_rag(name: str = "rag_retrieval"):
    """RAG 검색 트레이싱 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = get_langfuse()
            
            trace = langfuse.trace(name=name, tags=["rag"])
            
            query = kwargs.get("query", "")
            domain = kwargs.get("domain", "default")
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                langfuse.span(
                    trace=trace,
                    name="retrieval",
                    input={"query": query, "domain": domain},
                    metadata={
                        "duration_ms": (time.time() - start_time) * 1000,
                        "results_count": len(result) if result else 0
                    }
                )
                
                return result
            
            finally:
                trace.end()
        
        return wrapper
    return decorator
```

### 7.3 Prometheus Metrics

```python
# packages/agentic-ai-core/src/agentic_ai_core/observability/metrics/prometheus.py

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from functools import wraps
import time

class MetricsCollector:
    """Prometheus 메트릭 수집기"""
    
    def __init__(self, namespace: str = "agentic_ai"):
        self.namespace = namespace
        self.registry = CollectorRegistry()
        
        # 요청 메트릭
        self.request_count = Counter(
            f'{namespace}_requests_total',
            'Total requests',
            ['domain', 'agent', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            f'{namespace}_request_duration_seconds',
            'Request duration',
            ['domain', 'agent'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.active_requests = Gauge(
            f'{namespace}_active_requests',
            'Active requests',
            ['domain'],
            registry=self.registry
        )
        
        # LLM 메트릭
        self.llm_requests = Counter(
            f'{namespace}_llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_tokens = Counter(
            f'{namespace}_llm_tokens_total',
            'Total LLM tokens',
            ['provider', 'model', 'type'],
            registry=self.registry
        )
        
        self.llm_cost = Counter(
            f'{namespace}_llm_cost_usd_total',
            'Total LLM cost in USD',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.llm_latency = Histogram(
            f'{namespace}_llm_latency_seconds',
            'LLM response latency',
            ['provider', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # RAG 메트릭
        self.rag_retrievals = Counter(
            f'{namespace}_rag_retrievals_total',
            'Total RAG retrievals',
            ['domain', 'status'],
            registry=self.registry
        )
        
        self.rag_latency = Histogram(
            f'{namespace}_rag_latency_seconds',
            'RAG retrieval latency',
            ['domain'],
            registry=self.registry
        )
        
        # Tool 메트릭
        self.tool_executions = Counter(
            f'{namespace}_tool_executions_total',
            'Total tool executions',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        # 캐시 메트릭
        self.cache_hits = Counter(
            f'{namespace}_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            f'{namespace}_cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # 에러 메트릭
        self.errors = Counter(
            f'{namespace}_errors_total',
            'Total errors',
            ['component', 'error_type'],
            registry=self.registry
        )
    
    def track_request(self, domain: str, agent: str):
        """요청 추적 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                self.active_requests.labels(domain=domain).inc()
                
                start_time = time.time()
                status = "success"
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.errors.labels(
                        component=f"{domain}.{agent}",
                        error_type=type(e).__name__
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_duration.labels(domain=domain, agent=agent).observe(duration)
                    self.request_count.labels(domain=domain, agent=agent, status=status).inc()
                    self.active_requests.labels(domain=domain).dec()
            
            return wrapper
        return decorator
    
    def record_llm_call(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        latency: float,
        status: str = "success"
    ):
        """LLM 호출 기록"""
        self.llm_requests.labels(provider=provider, model=model, status=status).inc()
        self.llm_tokens.labels(provider=provider, model=model, type="input").inc(tokens_in)
        self.llm_tokens.labels(provider=provider, model=model, type="output").inc(tokens_out)
        self.llm_cost.labels(provider=provider, model=model).inc(cost)
        self.llm_latency.labels(provider=provider, model=model).observe(latency)
    
    def record_rag_retrieval(self, domain: str, latency: float, status: str = "success"):
        """RAG 검색 기록"""
        self.rag_retrievals.labels(domain=domain, status=status).inc()
        self.rag_latency.labels(domain=domain).observe(latency)
    
    def record_tool_execution(self, tool_name: str, status: str = "success"):
        """Tool 실행 기록"""
        self.tool_executions.labels(tool_name=tool_name, status=status).inc()
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """캐시 접근 기록"""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()


# 글로벌 메트릭 수집기
_metrics = None

def get_metrics() -> MetricsCollector:
    """메트릭 수집기 반환"""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
```

### 7.4 Structured Logger

```python
# packages/agentic-ai-core/src/agentic_ai_core/observability/logging/structured_logger.py

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

# 컨텍스트 변수
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
domain_var: ContextVar[str] = ContextVar('domain', default='')


class JSONFormatter(logging.Formatter):
    """JSON 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 컨텍스트 추가
        if request_id_var.get():
            log_data["request_id"] = request_id_var.get()
        if user_id_var.get():
            log_data["user_id"] = user_id_var.get()
        if domain_var.get():
            log_data["domain"] = domain_var.get()
        
        # 예외 정보
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 추가 필드
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


class StructuredLogger:
    """구조화된 로거"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if extra or kwargs:
            combined_extra = {**(extra or {}), **kwargs}
            self.logger.log(level, message, extra={'extra_data': combined_extra})
        else:
            self.logger.log(level, message)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


def setup_logging(level: str = "INFO", format_type: str = "json"):
    """로깅 설정"""
    log_level = getattr(logging, level.upper())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 새 핸들러 추가
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> StructuredLogger:
    """로거 인스턴스 반환"""
    return StructuredLogger(name)
```

---

## 8. 도메인별 서비스 구현 예시

### 8.1 쇼핑몰 서비스 - Customer Service Agent

```python
# services/ecommerce/src/ecommerce_service/agents/customer_service_agent.py

from agentic_ai_core.agents import BaseAgent, AgentConfig
from agentic_ai_core.observability.tracer import trace_agent
from ..tools import CustomerDBTool, OrderLookupTool, RefundProcessorTool

class CustomerServiceAgent(BaseAgent):
    """고객 서비스 전문 Agent"""
    
    NAME = "customer_service"
    DESCRIPTION = "고객 문의, 주문 조회, 환불 처리를 담당하는 Agent"
    CAPABILITIES = [
        "customer_inquiry",
        "order_lookup",
        "complaint_handling",
        "refund_processing"
    ]
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Tool 등록
        self.register_tools([
            CustomerDBTool(config.tools_config.get("customer_db", {})),
            OrderLookupTool(config.tools_config.get("order_lookup", {})),
            RefundProcessorTool(config.tools_config.get("refund_processor", {}))
        ])
        
        # 시스템 프롬프트
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        return """당신은 전문 이커머스 고객 서비스 매니저입니다.

역할:
- 고객 문의에 친절하고 정확하게 응답
- 주문 및 배송 상태 조회
- 환불/교환 요청 처리
- VIP 고객 특별 케어

주의사항:
- 항상 정중하고 전문적인 어조 유지
- 개인정보는 본인 확인 후에만 제공
- 환불 처리 전 정책 확인 필수
- 복잡한 문제는 담당자 연결 안내

사용 가능한 도구:
{tools}

고객의 요청을 분석하고, 필요한 도구를 사용하여 최선의 답변을 제공하세요.
"""
    
    @trace_agent(name="customer_service_execution")
    async def execute(self, task: str, context: dict) -> str:
        """고객 서비스 작업 실행"""
        
        # 시스템 프롬프트에 도구 정보 추가
        system_prompt = self.system_prompt.format(
            tools=self.get_tool_descriptions()
        )
        
        # 1. RAG로 관련 정책/FAQ 검색
        relevant_docs = await self.rag.retrieve(
            query=task,
            domain="ecommerce",
            top_k=3
        )
        
        # 2. 작업 계획 수립
        plan = await self._create_plan(task, relevant_docs)
        
        # 3. Tool 실행
        tool_results = await self._execute_tools(plan)
        
        # 4. 최종 응답 생성
        response = await self._generate_response(
            task=task,
            context=context,
            relevant_docs=relevant_docs,
            tool_results=tool_results
        )
        
        return response
```

### 8.2 쇼핑몰 서비스 - Tools

```python
# services/ecommerce/src/ecommerce_service/tools/customer_db_tool.py

from agentic_ai_core.tools import BaseTool
from typing import Dict, Any, Optional

class CustomerDBTool(BaseTool):
    """
    고객 정보 조회 도구
    
    Functions:
    - get_customer_by_email: 이메일로 고객 찾기
    - get_customer_by_id: ID로 고객 찾기
    - get_customer_tier: 고객 등급 조회
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.db_url = config.get("database_url")
        # 실제 구현에서는 DB 연결
    
    async def get_customer_by_email(self, email: str) -> Dict[str, Any]:
        """이메일로 고객 조회"""
        # 실제 구현: DB 쿼리
        # 예시 데이터 반환
        return {
            "customer_id": 12345,
            "name": "Erica Kim",
            "email": email,
            "phone": "010-1234-5678",
            "tier": "VIP",
            "signup_date": "2024-01-15",
            "total_orders": 24,
            "total_spent": 2040000
        }
    
    async def get_customer_by_id(self, customer_id: int) -> Dict[str, Any]:
        """ID로 고객 조회"""
        return {
            "customer_id": customer_id,
            "name": "Customer",
            "email": "customer@example.com",
            "tier": "regular"
        }
    
    async def get_customer_tier(self, customer_id: int) -> str:
        """고객 등급 조회"""
        customer = await self.get_customer_by_id(customer_id)
        return customer.get("tier", "regular")


# services/ecommerce/src/ecommerce_service/tools/order_lookup_tool.py

class OrderLookupTool(BaseTool):
    """
    주문 내역 조회 도구
    
    Functions:
    - get_order_history: 주문 이력 조회
    - get_order_frequency: 주문 빈도 분석
    - get_order_status: 주문 상태 확인
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
    
    async def get_order_history(
        self, 
        customer_id: int, 
        limit: int = 10
    ) -> list:
        """고객의 주문 이력"""
        # 예시 데이터
        return [
            {
                "order_id": "ORD-9901",
                "order_date": "2025-01-05",
                "total_amount": 89000,
                "status": "delivered"
            },
            {
                "order_id": "ORD-9845",
                "order_date": "2025-01-02",
                "total_amount": 125000,
                "status": "shipped"
            }
        ]
    
    async def get_order_frequency(self, customer_id: int) -> Dict[str, Any]:
        """주문 빈도 분석"""
        return {
            "total_orders": 24,
            "first_order_date": "2024-01-20",
            "last_order_date": "2025-01-05",
            "orders_per_month": 2.18,
            "average_order_value": 85000
        }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """주문 상태 확인"""
        return {
            "order_id": order_id,
            "status": "shipped",
            "tracking_number": "1234567890",
            "estimated_delivery": "2025-01-08"
        }


# services/ecommerce/src/ecommerce_service/tools/coupon_generator_tool.py

class CouponGeneratorTool(BaseTool):
    """
    쿠폰 생성 도구
    
    Functions:
    - create_coupon: 쿠폰 발급
    - validate_coupon: 쿠폰 유효성 확인
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
    
    async def create_coupon(
        self,
        customer_id: int,
        discount_type: str = "percentage",
        discount_value: float = 10,
        valid_days: int = 30,
        reason: str = ""
    ) -> Dict[str, Any]:
        """쿠폰 발급"""
        import uuid
        from datetime import datetime, timedelta
        
        coupon_code = f"VIP-JAN-2025-{uuid.uuid4().hex[:6].upper()}"
        
        return {
            "coupon_id": 78901,
            "code": coupon_code,
            "discount_type": discount_type,
            "discount_value": discount_value,
            "valid_until": (datetime.now() + timedelta(days=valid_days)).strftime("%Y-%m-%d"),
            "reason": reason
        }
    
    async def validate_coupon(self, coupon_code: str) -> Dict[str, Any]:
        """쿠폰 유효성 확인"""
        return {
            "valid": True,
            "discount_type": "percentage",
            "discount_value": 10,
            "remaining_uses": 1
        }
```

---

## 9. 설정 파일들

### 9.1 도메인 설정 (ecommerce)

```yaml
# services/ecommerce/config/domain.yaml

domain:
  name: "ecommerce"
  description: "쇼핑몰 운영 자동화"

persona:
  role: "전문 이커머스 매니저"
  tone: "친절하고 효율적"
  expertise:
    - "고객 문의 응대"
    - "재고 최적화"
    - "마케팅 분석"

orchestration:
  pattern: "supervisor"
  
agents:
  - name: "customer_service"
    class: "ecommerce_service.agents.CustomerServiceAgent"
    role: "고객 응대"
    capabilities:
      - "customer_inquiry"
      - "order_lookup"
      - "refund_processing"
  
  - name: "inventory_manager"
    class: "ecommerce_service.agents.InventoryManagerAgent"
    role: "재고 관리"
    capabilities:
      - "stock_check"
      - "reorder_suggestion"
  
  - name: "marketing_analyst"
    class: "ecommerce_service.agents.MarketingAnalystAgent"
    role: "마케팅 분석"
    capabilities:
      - "campaign_analysis"
      - "customer_segmentation"
```

### 9.2 LLM 설정

```yaml
# services/ecommerce/config/llm.yaml

llm_gateway:
  primary_provider: "local_vllm"
  
  fallback_providers:
    - "openai"
    - "anthropic"
  
  providers:
    - type: "local_vllm"
      base_url: "http://gpu-server:8000"
      model_name: "meta-llama/Llama-3-70b-Instruct"
      enabled: true
    
    - type: "openai"
      api_key: "${OPENAI_API_KEY}"
      default_model: "gpt-4o-mini"
      enabled: true
    
    - type: "anthropic"
      api_key: "${ANTHROPIC_API_KEY}"
      default_model: "claude-haiku-4-5-20251001"
      enabled: true
  
  routing:
    strategy: "cost"
    default_provider: "local_vllm"
  
  cache:
    enabled: true
    ttl: 3600
    use_redis: true
    redis_url: "${REDIS_URL}"
```

### 9.3 RAG 설정

```yaml
# services/ecommerce/config/rag.yaml

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  # or for local:
  # provider: "local"
  # model: "BAAI/bge-m3"

vector_store:
  type: "milvus"
  connection:
    host: "${MILVUS_HOST}"
    port: 19530
    user: "${MILVUS_USER}"
    password: "${MILVUS_PASSWORD}"
  collection:
    name: "ecommerce_knowledge"
    dimension: 1536
    index_type: "IVF_FLAT"
    metric_type: "L2"

chunking:
  strategy: "recursive"
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
  reranking:
    enabled: true
    model: "cross-encoder"
  query_rewriting:
    enabled: true
    methods:
      - "expansion"

sources:
  - type: "directory"
    path: "./knowledge/docs"
    file_types:
      - ".md"
      - ".txt"
      - ".pdf"
  
  - type: "database"
    connection: "${DATABASE_URL}"
    tables:
      - "products"
      - "categories"
      - "policies"
    update_frequency: "daily"
```

### 9.4 보안 설정

```yaml
# services/ecommerce/config/security/roles.yaml

roles:
  customer:
    description: "일반 고객"
    allowed_agents:
      - "customer_service"
    allowed_tools:
      - "order_lookup"
      - "product_search"
    data_scope: "own"
  
  merchant:
    description: "쇼핑몰 운영자"
    allowed_agents:
      - "customer_service"
      - "inventory_manager"
      - "marketing_analyst"
    allowed_tools:
      - "order_lookup"
      - "customer_db"
      - "inventory_checker"
      - "coupon_generator"
    data_scope: "all"
  
  admin:
    description: "시스템 관리자"
    allowed_agents:
      - "*"
    allowed_tools:
      - "*"
    data_scope: "all"
```

---

## 10. 인프라 및 배포

### 10.1 Docker Compose

```yaml
# infrastructure/docker/docker-compose.yml

version: '3.8'

services:
  # API 서비스
  api:
    build:
      context: ../../
      dockerfile: services/ecommerce/docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MILVUS_HOST=milvus
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agentic_ai
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
    depends_on:
      - postgres
      - redis
      - milvus
    restart: unless-stopped
  
  # PostgreSQL
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: agentic_ai
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  # Milvus
  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ALLOW_NONE_AUTHENTICATION=yes
    volumes:
      - milvus_etcd_data:/etcd
  
  milvus-minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - milvus_minio_data:/minio_data
    command: minio server /minio_data
  
  milvus:
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - milvus-etcd
      - milvus-minio

volumes:
  postgres_data:
  redis_data:
  milvus_data:
  milvus_etcd_data:
  milvus_minio_data:
```

### 10.2 모니터링 스택

```yaml
# infrastructure/docker/docker-compose.monitoring.yml

version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  
  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
  
  # Jaeger (트레이싱)
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "6831:6831/udp"  # Agent
      - "14268:14268"  # Collector
  
  # Langfuse (Self-hosted)
  langfuse:
    image: ghcr.io/langfuse/langfuse:latest
    ports:
      - "3001:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
      - NEXTAUTH_SECRET=your-secret
      - SALT=your-salt
    depends_on:
      - postgres
  
  # Attu (Milvus UI)
  attu:
    image: zilliz/attu:latest
    ports:
      - "3002:3000"
    environment:
      - MILVUS_URL=milvus:19530

volumes:
  prometheus_data:
  grafana_data:
```

---

## 11. 구현 로드맵

### Phase 1: MVP (Week 1-4)

```
Week 1-2: 코어 기반
├─ Day 1-2: 프로젝트 초기화 & 기본 구조
├─ Day 3-4: 설정 시스템 & 유틸리티
├─ Day 5-7: 데이터베이스 & Milvus 연결
└─ Day 8-10: LLM Gateway (OpenAI)

Week 3-4: RAG + Agent
├─ Day 11-13: 임베딩 & Milvus 통합
├─ Day 14-16: Retriever & Chunker
├─ Day 17-20: 인덱싱 파이프라인
├─ Day 21-23: Base Agent & Tool
├─ Day 24-26: Orchestrator (Supervisor)
└─ Day 27-30: 첫 번째 도메인 Agent
```

### Phase 2: 보안 & 관찰성 (Week 5-6)

```
Week 5: 보안
├─ Day 31-33: RBAC 구현
├─ Day 34-35: PBAC 추가
└─ Day 36-37: 감사 로깅

Week 6: 관찰성
├─ Day 38-39: Langfuse 통합
├─ Day 40-41: Prometheus 메트릭
└─ Day 42-44: 대시보드 구성
```

### Phase 3: 프로덕션 준비 (Week 7-8)

```
Week 7: API & 테스트
├─ Day 45-47: FastAPI 엔드포인트
├─ Day 48-49: 통합 테스트
└─ Day 50-51: E2E 테스트

Week 8: 배포
├─ Day 52-53: Docker 이미지
├─ Day 54-55: K8s 매니페스트
└─ Day 56-58: CI/CD 파이프라인
```

---

## 시작하기

```bash
# 1. 프로젝트 클론
git clone https://github.com/your-org/agentic-ai-platform.git
cd agentic-ai-platform

# 2. 가상환경 생성
make setup
source venv/bin/activate

# 3. 코어 패키지 설치
make install-core

# 4. 환경 변수 설정
cp .env.example .env
vim .env  # API 키 등 설정

# 5. 인프라 시작
make infra-up

# 6. 데이터베이스 마이그레이션
make migrate-up

# 7. 문서 인덱싱
python scripts/indexing/index_documents.py --domain ecommerce

# 8. 서비스 시작
cd services/ecommerce
uvicorn api.main:app --reload

# 9. 테스트
make test-all
```

---

## 요약

이 가이드는 다음을 제공합니다:

- **완전한 프로젝트 구조**: 코어 패키지 + 도메인 서비스 분리
- **Orchestrator 구현**: Supervisor, Hierarchy, Collaborative, Sequential 패턴
- **Agent 시스템**: BaseAgent, Tool Registry, Security Wrapper
- **RAG 파이프라인**: Embedder, Milvus, Chunker, Retriever
- **LLM Gateway**: 멀티 프로바이더, 라우팅, 폴백, 캐싱
- **보안**: RBAC + PBAC 하이브리드, JWT, 감사 로깅
- **Observability**: Langfuse, Prometheus, 구조화 로깅
- **인프라**: Docker Compose, K8s, CI/CD

**핵심 원칙**: "AI 엔진은 똑같은데, 도메인 지식만 갈아끼우면 되는 구조"