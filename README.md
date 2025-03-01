```
import runpod, base64, uuid, os

runpod.api_key = 'api_key'
endpoint = runpod.Endpoint("runpod_pod_id")

lora_name = 'dustin moskovitz'

run_request = endpoint.run(
    {
      "input": {
          "positive_prompt": f"""a stunning portrait of {lora_name}""",
        "seed": 0,
        "steps": 20,
        "guidance": 3.5,
        "lora_url": "lora_download_url",
        "lora_strength_model": 1,
        "lora_strength_clip": 1,
        "sampler_name": "euler",
        "scheduler": "simple",
        "width": 896,
        "height": 1024
      }
    }
)

while True:
    status = run_request.status()
    if status == 'FAILED':
        print(run_request.output())
        break
    elif status == 'COMPLETED':
        base64_string = run_request.output()['image']
        image_data = base64.b64decode(base64_string)
        unique_filename = f"{uuid.uuid4().hex}.png"
        output_directory = "images"
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, unique_filename)
        with open(output_path, "wb") as image_file:
            image_file.write(image_data)
        print(f"Image saved as {output_path}")
        break
    else:
        print(status)
        continue
```
