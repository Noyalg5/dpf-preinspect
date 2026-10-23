"""Request and response schemas for the inspection API (pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Route = Literal["PROCEED", "EXTENDED_CLEAN", "QUARANTINE", "REFER"]
Outcome = Literal["PASS", "REWORK", "SCRAP"]
DISCLAIMER = (
    "Research prototype trained on synthetic data only. Not validated for production "
    "use; every QUARANTINE requires confirmation by a person."
)


class FlowCurve(BaseModel):
    """Incoming flow-bench measurement: pressure drop at each flow rate."""

    q_m3s: list[float] = Field(min_length=3, description="Volumetric flow rates, m3/s")
    dp_kpa: list[float | None] = Field(description="Pressure drop, kPa; null for a dropout")

    @model_validator(mode="after")
    def _same_length(self) -> FlowCurve:
        if len(self.q_m3s) != len(self.dp_kpa):
            raise ValueError(
                f"q_m3s has {len(self.q_m3s)} values but dp_kpa has "
                f"{len(self.dp_kpa)}; give one pressure drop per flow rate"
            )
        if any(b <= a for a, b in zip(self.q_m3s, self.q_m3s[1:], strict=False)):
            raise ValueError("q_m3s must be strictly increasing")
        return self


class Metadata(BaseModel):
    """Service history and part data recorded at goods-in."""

    material: str = Field(examples=["cordierite", "sic"])
    cell_density_cpsi: float = Field(gt=0)
    diameter_mm: float = Field(gt=0)
    length_mm: float = Field(gt=0)
    application: str = Field(examples=["urban_bus", "regional_haulage"])
    operating_hours: float = Field(ge=0)
    active_regen_count: int = Field(ge=0)
    max_regen_temp_c: float | None = Field(default=None, description="Often unavailable")
    weight_gain_g: float
    previous_cleans: int = Field(ge=0)


class InspectRequest(BaseModel):
    """Everything except the two images, which are sent as files."""

    unit_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    metadata: Metadata
    flow: FlowCurve
    lighting: dict[Literal["inlet", "outlet"], str] = Field(
        description="Station lighting mode used for each face, e.g. diffuse"
    )
    include_maps: bool = False

    @field_validator("lighting")
    @classmethod
    def _both_views(cls, v: dict[str, str]) -> dict[str, str]:
        missing = {"inlet", "outlet"} - set(v)
        if missing:
            raise ValueError(f"lighting mode missing for: {', '.join(sorted(missing))}")
        return v


class DpInterval(BaseModel):
    """Predicted post-clean ΔP ratio with its conformal interval."""

    prediction: float
    lower: float
    upper: float
    coverage: float


class AnomalyInfo(BaseModel):
    """Anomaly-guard scores."""

    score: float
    unexplained: float
    threshold: float


class InspectResponse(BaseModel):
    """Result of one inspection, as logged to the audit trail."""

    inspection_id: int
    unit_id: str
    valid: bool
    input_issues: list[str]
    route: Route
    reason: str
    rationale: str
    probabilities: dict[Outcome, float]
    prediction_set: list[Outcome]
    dp_ratio: DpInterval
    anomaly: AnomalyInfo
    defects: dict[str, dict[str, float | None]]
    maps: dict[str, dict[str, str]] | None = None
    model_version: str
    data_version: str | None
    code_version: str
    deployed_model: str
    disclaimer: str = DISCLAIMER


class DecisionRequest(BaseModel):
    """An operator accepting or overriding the system's route."""

    inspection_id: int = Field(ge=1)
    unit_id: str
    accepted: bool
    route: Route
    reason: str | None = None
    operator: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _override_needs_reason(self) -> DecisionRequest:
        if not self.accepted and (self.reason is None or len(self.reason.strip()) < 5):
            raise ValueError(
                "an override needs a reason of at least 5 characters so the "
                "audit trail can explain it"
            )
        return self


class ResultRequest(BaseModel):
    """Downstream test result, reported once the unit has been cleaned and tested."""

    unit_id: str
    disposition: Outcome
    dp_ratio_post_clean: float | None = None
