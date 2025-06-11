import uvicorn
import os
from api.fast import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.fast:app", host="0.0.0.0", port=port, reload=True)
