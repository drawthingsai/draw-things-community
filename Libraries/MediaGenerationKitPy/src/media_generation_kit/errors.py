class MediaGenerationKitError(Exception):
    """Base error for MediaGenerationKitPy."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.message:
            return f"{self.code}: {self.message}"
        return self.code

    @classmethod
    def generation_failed(cls, message: str = "") -> "MediaGenerationKitError":
        return cls("generationFailed", message)

    @classmethod
    def unresolved_model_reference(
        cls, query: str, suggestions: list[str] | None = None
    ) -> "MediaGenerationKitError":
        suggestions = suggestions or []
        if not suggestions:
            return cls("unresolvedModelReference", f"Could not resolve model reference: {query}")
        rendered = "\n".join(f"  - {item}" for item in suggestions)
        return cls(
            "unresolvedModelReference",
            f"Could not resolve model reference '{query}'.\nClosest matches:\n{rendered}",
        )

    @classmethod
    def model_not_found_in_catalog(cls, model: str) -> "MediaGenerationKitError":
        return cls("modelNotFoundInCatalog", f"Model not found in catalog: {model}")

    @classmethod
    def model_not_found_on_remote(cls, model: str) -> "MediaGenerationKitError":
        return cls("modelNotFoundOnRemote", f"Model not found on remote: {model}")

    @classmethod
    def remote_not_configured(cls) -> "MediaGenerationKitError":
        return cls("remoteNotConfigured", "Remote not configured")

    @classmethod
    def cancelled(cls) -> "MediaGenerationKitError":
        return cls("cancelled", "Generation cancelled")
