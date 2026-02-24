from typing import Any, Callable, Optional

from pydantic import BaseModel, PrivateAttr



class ImageResponse(str):
    """
    A response from a tool that includes an image.
    Behaves like a string (returning the text content) but carries extra image data.
    """
    def __new__(cls, text_content, image_data_base64=None, image_path=None):
        instance = super().__new__(cls, text_content)
        instance.image_data_base64 = image_data_base64
        instance.image_path = image_path
        return instance

def type_to_string(typ):
    match typ():
        case str():
            return "string"
        case float():
            return "number"
        case int():
            return "integer"
        case dict():
            return "object"
        case set():
            return "array"
        case list():
            return "array"



class FunctionParameter(BaseModel):
    name: str
    param_type: type
    param_description: str
    required: bool = True
    enum_values: Optional[Any] = None

class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: list[FunctionParameter]
    function: Callable
    _bound_args: PrivateAttr() = None

