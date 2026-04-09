import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# List of webpages to crawl
urls = [
   "https://www.dfrobot.com.cn/goods-4108.html",
   "https://www.dfrobot.com.cn/goods-4097.html",
   "https://www.dfrobot.com.cn/goods-149.html"
]

# Create a folder to save images
os.makedirs("images", exist_ok=True)

for url in urls:
    print(f"\nCrawling: {url}")
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get all image tags
        img_tags = soup.find_all("img")

        for img in img_tags:
            img_url = img.get("src")
            if not img_url:
                continue

            # Handle relative URLs
            full_url = urljoin(url, img_url)
            img_name = os.path.basename(full_url.split("?")[0])

            #Save the image
            try:
                if "png" in img_name.lower() or "jpg" in img_name.lower() or "jpeg" in img_name.lower():
                    img_data = requests.get(full_url).content
                    with open(os.path.join("images", img_name), "wb") as f:
                        f.write(img_data)
                    print(f"Downloaded {img_name}")
                else:
                    print(f"Got {img_name}. Skipping...")
            except Exception as e:
                print(f"Failed to download {full_url}: {e}")

    except Exception as e:
        print(f"Failed to crawl {url}: {e}")
