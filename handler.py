import runpod

from app.api import handler as _handler

runpod.serverless.start({"handler": _handler})
