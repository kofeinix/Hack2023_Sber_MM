import redis
import logging
logger = logging.getLogger(__name__)


def get_redis():
    logger.debug('Initializing Redis connection')
    try:
        conn = redis.Redis(
        host='localhost', port=6379, health_check_interval=10,
        socket_timeout=10, socket_keepalive=True,
        socket_connect_timeout=10, retry_on_timeout=True
        )
        return conn
    except Exception as e:
        logger.error(f'Could not connect to Redis due to {e}')