
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import requests
from datetime import datetime

# Update this URL to your server's URL if hosted remotely
API_URL = "http://localhost:8000/predict"


def send_generate_request(image, p1, p2, model, alpha):
    response = requests.post(API_URL, files={"image": image, "p1": (None, p1), "p2": (None, p2),
                                             "alpha": (None, alpha), "model": (None, model)})

    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").lower()
        filename = f"output-{timestamp}.png"

        with open(filename, "wb") as output_file:
            output_file.write(response.content)

        print(f"Image saved to {filename}")
    else:
        print(f"Error: Response with status code {response.status_code} - {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send text & image to server and receive a response.")
    parser.add_argument("--image", required=True, help="URL for the image file.")
    parser.add_argument("--p1", required=False, help="Single Point input in '(x,y)' format.")
    parser.add_argument("--p2", required=False, help="Point2 '(x1,y1)' for box input.")
    parser.add_argument("--model", required=False, help="Model used for inference", default="med_sam-vit_b")
    parser.add_argument("--alpha", required=False, help="Transparency mask between 0-1.", default="0.5")

    args = parser.parse_args()

    send_generate_request(args.image, args.p1, args.p2, args.model, args.alpha)