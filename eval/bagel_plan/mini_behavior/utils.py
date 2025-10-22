def in_bounds_rc(r: int, c: int, H: int, W: int) -> bool:
    return 0 <= r < H and 0 <= c < W

def can_interact(agent_position, object_position):
    """
    判断物品是否可以被拾取
    
    Args:
        agent_position: (x, y, direction) - agent的二维坐标和方向
        object_position: (x, y) - object的二维坐标
        方向编码: 0=朝右, 1=朝下, 2=朝左, 3=朝上
    
    Returns:
        bool: 是否可以拾取
    """
    agent_x, agent_y, direction = agent_position
    obj_x, obj_y = object_position
    
    # 检查是否在相邻格子（曼哈顿距离为1）
    manhattan_distance = abs(agent_x - obj_x) + abs(agent_y - obj_y)
    if manhattan_distance != 1:
        return False
    
    # 检查agent是否朝向object
    # 计算从agent到object的方向向量
    dx = obj_x - agent_x
    dy = obj_y - agent_y
    
    # 根据agent的方向检查是否朝向object
    if direction == 0:  # 朝右
        return dx == 0 and dy == 1
    elif direction == 1:  # 朝下
        return dx == 1 and dy == 0
    elif direction == 2:  # 朝左
        return dx == 0 and dy == -1
    elif direction == 3:  # 朝上
        return dx == -1 and dy == 0
    else:
        return False
    

def can_move_forward(H, W, agent_position, object_position, table_position):
    """
    判断agent是否可以向前移动一格
    
    Args:
        agent_position: (x, y, direction) - agent的二维坐标和方向
        object_position: (x, y) - object的二维坐标
        table_position: (x, y) - table的二维坐标
        方向编码: 0=朝右, 1=朝下, 2=朝左, 3=朝上
    
    Returns:
        bool: 是否可以向前移动
    """
    agent_x, agent_y, direction = agent_position
    obj_x, obj_y = object_position
    table_x, table_y = table_position
    
    # 根据方向计算移动后的位置
    if direction == 0:  # 朝右
        next_x, next_y = agent_x, agent_y + 1
    elif direction == 1:  # 朝下
        next_x, next_y = agent_x + 1, agent_y
    elif direction == 2:  # 朝左
        next_x, next_y = agent_x, agent_y - 1
    elif direction == 3:  # 朝上
        next_x, next_y = agent_x - 1, agent_y
    else:
        return False  # 无效方向
    
    if not in_bounds_rc(next_x, next_y, H, W):
        return False
    
    # 检查移动后是否与object或table位置重叠
    if (next_x, next_y) == (obj_x, obj_y) or (next_x, next_y) == (table_x, table_y):
        return False
    
    return True