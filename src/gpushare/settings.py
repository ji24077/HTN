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

    # Demo knobs
    rho: float = 0.05                                # raise to 0.2 for the stage


settings = Settings()
