"""Serve the buildless UI without mixing old scripts with newly updated HTML."""

from fastapi.staticfiles import StaticFiles


class FrontendFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers):
        # A cached module can lack handlers for controls already present in new HTML.
        # This small local demo always serves the current frontend files together.
        return False

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response
