import base64
import os
import glob
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from heracles.query_interface import Neo4jWrapper
from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_agents.tool_interface import ImageResponse
from heracles_agents.tool_registry import ToolRegistry, register_tool


class BoundingBox(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int


def get_image_from_node(node_symbol, db):
    # Query for image folder
    res = db.query(f"MATCH (n:Object {{nodeSymbol: '{node_symbol}'}}) RETURN n.image_folder")
    if not res:
        return None, "Node not found or no image folder"
    
    image_folder = res[0]['n.image_folder']
    if not image_folder or not os.path.exists(image_folder):
        return None, f"Image folder not found: {image_folder}"
        
    # Find an image (prioritize rgb)
    # Common patterns: frame_..._rgb.jpg, or just rgb.png
    extensions = ['*_rgb.jpg', '*_rgb.png', '*_rgb.jpeg', '*.jpg', '*.png', '*.jpeg']
    candidate_images = []
    
    for ext in extensions:
        found = glob.glob(os.path.join(image_folder, ext))
        # Filter out depth/mask if using generic wildcards
        for f in found:
            fname = os.path.basename(f)
            if "depth" in fname or "mask" in fname:
                continue
            candidate_images.append(f)
        
        # If we found explicit rgb images, stop there
        if candidate_images and "rgb" in ext:
             break
             
    if not candidate_images:
        return None, f"No suitable RGB images found in {image_folder}"
        
    # Pick the first one
    return candidate_images[0], None


def vlm_relocalize(node_symbol: str, prompt: str, dsgdb_conf: HeraclesDsgInterface = None):
    if dsgdb_conf is None:
        raise ValueError("vlm_relocalize called with dsgdb_conf=None")
        
    # 1. Connect to DB and get image path
    with Neo4jWrapper(
        dsgdb_conf.uri,
        (dsgdb_conf.username.get_secret_value(), dsgdb_conf.password.get_secret_value()),
        atomic_queries=True,
        print_profiles=False,
    ) as db:
        image_path, error = get_image_from_node(node_symbol, db)
        
    if error:
        return f"Error locating image: {error}"
        
    # 2. Load image and convert to base64
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error loading image {image_path}: {e}"

    # 3. Call VLM (Generic)
    api_key = os.environ.get("VLM_API_KEY", os.environ.get("QWEN_API_KEY", "EMPTY"))
    base_url = os.environ.get("VLM_BASE_URL", os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"))
    model_name = os.environ.get("VLM_MODEL_NAME", "qwen/Qwen2-VL-72B-Instruct")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Construct Chat Completion
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant for localizing objects. Return the bounding box as a valid JSON object matching this schema: {json.dumps(BoundingBox.model_json_schema())}. Do not wrap in markdown blocks."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Locate {prompt} in this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        content = response.choices[0].message.content
        
        # 4. Validate Output
        try:
            # First try raw json
            bbox = BoundingBox.model_validate_json(content)
        except ValidationError:
            try:
                # If wrapped in markdown ```json ... ```, extract it
                # Simple extraction, looking for first { and last }
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    bbox = BoundingBox.model_validate_json(json_str)
                else:
                    raise ValueError("Could not find JSON object")
            except Exception as e:
                return f"Error validating VLM output: {e}. Raw output: {content}"

        result_text = f"Localized Bounding Box: {bbox.model_dump_json()}"

    except Exception as e:
        return f"Error calling VLM: {e}"

    # Return Result wrapped in ImageResponse
    return ImageResponse(
        text_content=result_text,
        image_data_base64=base64_image,
        image_path=image_path
    )


register_tool(vlm_relocalize)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
