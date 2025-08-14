import functools

def with_notice(operation_name):
    """MCP操作通知装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 对于异步函数
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"MCP操作 '{operation_name}' 失败: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 对于同步函数
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"MCP操作 '{operation_name}' 失败: {e}")
                raise
        
        # 根据函数类型返回对应的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator