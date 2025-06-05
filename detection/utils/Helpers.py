def getNodeNumber(node_name: str):
    number = ''.join(filter(str.isdigit, node_name))
    return int(number) if number else None
