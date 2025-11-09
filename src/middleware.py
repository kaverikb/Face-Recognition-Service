from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FRS")

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Log request
        start_time = time.time()
        
        logger.info(f"→ {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(f"← {request.method} {request.url.path} - "
                       f"{response.status_code} ({process_time*1000:.2f}ms)")
            
            return response
        
        except Exception as e:
            logger.error(f"✗ {request.method} {request.url.path} - Error: {str(e)}")
            raise