from typing import TypedDict

from pydantic import BaseModel, Field, SecretStr, field_validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class OracleConfigDict(TypedDict):
    user: str
    password: str
    host: str
    port: int
    service_name: str
    sysdba: bool
    collection_name: str


class OracleConfig(DBConfig):
    user_name: SecretStr = SecretStr("vdbbench_user")
    password: SecretStr = SecretStr("vdbbench_pass")
    host: str = "localhost"
    port: int = 1521
    service_name: str = "FREEPDB1"
    sysdba: bool = False
    table_name: str = "vdbbench_oracle_test"

    def to_dict(self) -> OracleConfigDict:
        user_str = self.user_name.get_secret_value() if isinstance(self.user_name, SecretStr) else str(self.user_name)
        pwd_str = self.password.get_secret_value() if isinstance(self.password, SecretStr) else str(self.password)
        return {
            "user": user_str,
            "password": pwd_str,
            "host": self.host,
            "port": self.port,
            "service_name": self.service_name,
            "sysdba": self.sysdba,
            "collection_name": self.table_name,
        }


def parse_oracle_metric(metric_type: MetricType | None) -> str:
    if metric_type is None:
        return "COSINE"
    if metric_type == MetricType.L2:
        return "EUCLIDEAN"
    if metric_type == MetricType.COSINE:
        return "COSINE"
    if metric_type in {MetricType.IP, MetricType.DP}:
        return "DOT"
    msg = f"Unsupported metric type for Oracle AI Vector Search: {metric_type}"
    raise ValueError(msg)


class OracleHNSWConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    neighbors: int = Field(32, ge=2, le=2048)
    ef_construction: int = Field(200, ge=1, le=65535)
    index_target_accuracy: int = Field(95, ge=1, le=100)
    search_target_accuracy: int = Field(95, ge=1, le=100)
    # Fixed-effort search: WITH TARGET ACCURACY PARAMETERS (EFSEARCH n) instead
    # of the adaptive accuracy target — the direct analogue of pgvector's
    # ef_search, for apples-to-apples cross-engine comparisons. None keeps the
    # adaptive TARGET ACCURACY behaviour.
    ef_search: int | None = Field(None, ge=0, le=65535)
    create_index_after_load: bool = True
    create_index_before_load: bool = False

    @field_validator("ef_search")
    @classmethod
    def _zero_means_adaptive(cls, v: int | None) -> int | None:
        # 0 is the UI/CLI sentinel for "keep adaptive TARGET ACCURACY"
        return None if v == 0 else v

    def index_param(self) -> dict:
        return {
            "index_type": IndexType.HNSW.value,
            "neighbors": self.neighbors,
            "ef_construction": self.ef_construction,
            "index_target_accuracy": self.index_target_accuracy,
            "metric": parse_oracle_metric(self.metric_type),
        }

    def search_param(self) -> dict:
        param = {
            "search_target_accuracy": self.search_target_accuracy,
            "metric": parse_oracle_metric(self.metric_type),
        }
        if self.ef_search is not None:
            param["ef_search"] = self.ef_search
        return param


class OracleIVFConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    neighbor_partitions: int = Field(1024, ge=1, le=65535)
    samples_per_partition: int = Field(10, ge=1, le=1000)
    min_vectors_per_partition: int = Field(5, ge=1, le=1000)
    index_target_accuracy: int = Field(90, ge=1, le=100)
    search_target_accuracy: int = Field(90, ge=1, le=100)
    # Fixed-effort search: WITH TARGET ACCURACY PARAMETERS (NEIGHBOR PARTITION
    # PROBES n) — the direct analogue of pgvector's probes. None keeps the
    # adaptive TARGET ACCURACY behaviour.
    neighbor_partition_probes: int | None = Field(None, ge=0, le=65535)
    create_index_after_load: bool = True
    create_index_before_load: bool = False

    @field_validator("neighbor_partition_probes")
    @classmethod
    def _zero_means_adaptive(cls, v: int | None) -> int | None:
        # 0 is the UI/CLI sentinel for "keep adaptive TARGET ACCURACY"
        return None if v == 0 else v

    def index_param(self) -> dict:
        return {
            "index_type": IndexType.IVFFlat.value,
            "neighbor_partitions": self.neighbor_partitions,
            "samples_per_partition": self.samples_per_partition,
            "min_vectors_per_partition": self.min_vectors_per_partition,
            "index_target_accuracy": self.index_target_accuracy,
            "metric": parse_oracle_metric(self.metric_type),
        }

    def search_param(self) -> dict:
        param = {
            "search_target_accuracy": self.search_target_accuracy,
            "metric": parse_oracle_metric(self.metric_type),
        }
        if self.neighbor_partition_probes is not None:
            param["neighbor_partition_probes"] = self.neighbor_partition_probes
        return param


class OracleFlatConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    create_index_after_load: bool = False
    create_index_before_load: bool = False

    def index_param(self) -> dict:
        return {
            "index_type": IndexType.Flat.value,
            "metric": parse_oracle_metric(self.metric_type),
        }

    def search_param(self) -> dict:
        return {
            "metric": parse_oracle_metric(self.metric_type),
        }
