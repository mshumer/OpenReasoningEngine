def validate_conversation(history):
    """
    Before generating the final response, ensure all tool calls have responses.
    If a tool call doesn't have a response, include it in the message content instead.
    """
    tool_call_ids = set()
    tool_response_ids = set()
    
    for message in history:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                tool_call_ids.add(tool_call["id"])
        elif message.get("role") == "tool":
            tool_response_ids.add(message["tool_call_id"])
    
    # If there are unmatched tool calls, convert them to content
    if tool_call_ids != tool_response_ids:
        filtered_history = []
        
        for message in history:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                message_copy = message.copy()
                content = message_copy.get("content", "")
                
                # Convert unmatched tool calls to content
                for tool_call in message_copy["tool_calls"]:
                    if tool_call["id"] not in tool_response_ids:
                        tool_content = f"\nTool Call: {tool_call['function']['name']}({tool_call['function']['arguments']})"
                        content = (content + tool_content) if content else tool_content
                
                # Only keep matched tool calls
                message_copy["tool_calls"] = [tc for tc in message_copy["tool_calls"] 
                                            if tc["id"] in tool_response_ids]
                message_copy["content"] = content
                
                filtered_history.append(message_copy)
            else:
                filtered_history.append(message)
                
        return filtered_history
    return history