"""Shared config. Read from env / .env so nobody hardcodes an IP at 3am."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GPUSHARE_", extra="ignore")

    # Everyone
    server_url: str = "http://100.64.0.1:8000"      # Tailscale 100.x of the server box
    ws_url: str = "ws://100.64.0.1:8000"

    # Worker (Jack)
    worker_id: str = "w1"
    owner: str = "unknown"
    credits_per_hour: float = 1.0

    # Trainer (Ethan)
    master_addr: str = "100.64.0.1"                  # Tailscale IP, not a 192.168.x
    master_port: int = 29500
    gloo_iface: str = "tailscale0"

    # Agent LLM layer (Ji). Provider-agnostic: any OpenAI-compatible endpoint.
    # Baseten by default because OpenAI's balance is $0; switching back is one
    # env var. API keys are NOT here on purpose - the SDK reads them straight
    # from the environment, so a secret never lands in a model that gets logged.
    llm_base_url: str = "https://inference.baseten.co/v1"
    llm_model: str = "zai-org/GLM-5.3-Flash"
    llm_enabled: bool = True
    llm_timeout_s: float = 10.0                      # stage patience, not server patience

    # Demo knobs
    rho: float = 0.05                                # raise to 0.2 for the stage


settings = Settings()
