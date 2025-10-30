from io import BytesIO
from openai import OpenAI
from PIL import Image
import base64
import pandas as pd
from pprint import pprint
import requests
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import json
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import os


API_KEY = "your_openai_api_key_here"
IMAGE_PATH = "test_image.jpg"



PROMPT_TEXT_v1 = """
    You are a fashion normalizer. From the image, list all visible clothing items.
Return STRICT JSON (no prose) with this schema:
{
    "items": [
        {
        "category":"<t-shirt|shirt|blouse|hoodie|sweater|jacket|coat|blazer|dress|skirt|jeans|trousers|shorts| "color_primary": "<black|white|navy|blue|light blue|grey|beige|brown|green|red>",
        "pattern": "<solid|striped|checked|floral|graphic|textured|other>",
        "material": ["<cotton|wool|denim|leather|synthetic|linen|silk|blend>"],
        "details": ["<up to 3 short phrases, e.g., notch lapel, crewneck, two-button, if shirt mention sleeve length>"],
        "confidence": 0.0-1.0
        }
    ]
}
Rules:
- Use only the listed vocabulary. If unsure, choose the closest and lower confidence.
- Keep fields short; no bounding boxes; no explanations.

"""

PROMPT_TEXT_v2 = """
    You are a fashion normalizer. From the image, list all visible clothing items.
    The response you will give will be used to search for similar items in a database of clothings in order to find the most similar matches
Return STRICT JSON (no prose) with this schema:
{
    "items": [
        {
        "category":"<Topwear|Bottomwear|Innerwear|Saree|Dress|Loungewear|Nightwear|Socks>",
        "article_type": "<Shirts|Jeans|Track Pants|Tshirts|Tops|Bra|Sweatshirts|Kurtas|Waistcoat|Shorts|Briefs|Sarees|Innerwear Vests|Rain Jacket|Dresses|Night suits|Skirts|Blazers|Kurta Sets|Shrug|Trousers|Camisoles|Boxers|Dupatta|Capris|Bath Robe|Tunics|Jackets|Trunk|Sweaters|Tracksuits|Swimwear|Nightdress|Baby Dolls|Leggings|Kurtis|Jumpsuit|Suspenders|Robe|Salwar and Dupatta|Patiala|Stockings|Tights|Churidar|Lounge Tshirts|Lounge Shorts|Shapewear|Nehru Jackets|Salwar|Jeggings|Rompers|Booties|Lehenga Choli|Clothing Set|Belts|Rain Trousers|Suits>",
        "color_primary": "<Navy Blue|Blue|Black|Grey|Green|Purple|White|Beige|Brown|Pink|Maroon|Red|Off White|Coffee Brown|Yellow|Charcoal|Multi|Magenta|Orange|Sea Green|Cream|Peach|Olive|Burgundy|Grey Melange|Rust|Rose|Lime Green|Teal|Khaki|Lavender|Mustard|Skin|Turquoise Blue|Nude|Mauve|Mushroom Brown|nan|Tan|Gold|Taupe|Silver|Fluorescent Green>",
        "color_primary_simple": "<Blue|Black|Grey|Green|Purple|White|Brown|Pink|Red|Yellow|Orange|Gold|Silver>",
        "pattern": "<solid|striped|checked|floral|graphic|textured|other>",
        "product_display_name": "<what would the product be called if listed on a store>",
        "material": ["<cotton|wool|denim|leather|synthetic|linen|silk|blend>"],
        "details": ["<up to 5 short phrases, e.g., notch lapel, crewneck, two-button, if shirt mention sleeve length, , gender of the item (male/female)>"],
        "confidence": 0.0-1.0
        }
    ]
}
Rules:
- Use only the listed vocabulary except for product_display_name. If unsure, choose the closest and lower confidence.
- Keep fields short; no bounding boxes; no explanations.
- Some example of product display names: 
    -Turtle Check Men Navy Blue Shirt, Manchester United Men Solid Black Track Pants, Inkfruit Mens Chain Reaction T-shirt, Tantra Women Printed Peach T-shirt, Sepia Women Blue Printed Top

"""

PROMPT_FINAL_v1 = """
    You are a fashion expert. You are given an image of a model.
    
    You are furthermore given several clothing items, You are provided the item name, product_id and the image of the item
    Sort these items by which most closely resembles the one worn by the model 
    
    
Return STRICT JSON (no prose) with this schema:
{
    "items": [
        {
        "product_id":"<product_id>",
        "product_name": "<product_name>",
        "order": "<order in similarity, 1 being most similar>"
        }
    ]
}

Rules:
- Keep fields short; no bounding boxes; no explanations.

The following is the product names and ids, attatched will also be the images:




"""

def resize_base64_image(base64_string, target_size=(200, 200), output_format='PNG'):
    """
    Decodes a base64 image, resizes it to the target size, and re-encodes it.

    :param base64_string: The base64-encoded image string (without the data URL prefix).
    :param target_size: A tuple (width, height) for the new size.
    :param output_format: The image format to use for re-encoding (e.g., 'PNG', 'JPEG').
    :return: The new base64-encoded image string.
    """
    # 1. Decode the base64 string to bytes
    img_data = base64.b64decode(base64_string)

    # 2. Load the image from bytes using an in-memory buffer (BytesIO)
    img_buffer = io.BytesIO(img_data)
    img = Image.open(img_buffer)

    # 3. Resize the image (Image.LANCZOS is a high-quality downsampling filter)
    # The 'resize' method changes the aspect ratio to fit the exact size.
    resized_img = img.resize(target_size, Image.LANCZOS)

    # 4. Save the resized image back to a new in-memory buffer
    output_buffer = io.BytesIO()
    
    # You must specify the output format when saving to bytes
    resized_img.save(output_buffer, format=output_format)

    # 5. Encode the new raw bytes back to a base64 string
    new_base64_bytes = base64.b64encode(output_buffer.getvalue())
    new_base64_string = new_base64_bytes.decode('utf-8')

    return new_base64_string


def read_images_csv(IMAGE_CSV_PATH="images2.csv"):
    df = pd.read_csv(IMAGE_CSV_PATH, on_bad_lines="skip")
    images_dict = df.to_dict(orient='records')
    
    # convert records to dict where key is 'filename' and value is 'image_url'
    images_dict = {item['filename']: item['link'] for item in images_dict}

    return images_dict

def get_llm_prediction_for_image(image_path=IMAGE_PATH):
    client = OpenAI(api_key=API_KEY)


    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")


    response = client.chat.completions.create(
        model="gpt-5-nano",
        store= True,
        top_p=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT_v2},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ],
            }
        ],
    )

    return response.choices[0].message.content


## Setup Knowledge base
def setup_knowledge_base_for_rag(model_name):
    filename = model_name+"_apparel_knowledge_base_rag.pkl"
    
    ## if Cache exists read it
    if os.path.exists(filename):
        print("Loading From Cache")
        df = pd.read_pickle(filename)
        return df
    model = SentenceTransformer(model_name) 

    df = pd.read_csv("styles.csv", on_bad_lines="skip")
    apparel_df = df[df['masterCategory'] == 'Apparel']
    apparel_df.reset_index(drop=True, inplace=True)
    tqdm.pandas(desc='My bar!')
    apparel_df['embedding'] = apparel_df['productDisplayName'].progress_apply(lambda x: model.encode(x).tolist())

    # Cache based on model type + dataset name as the filename
    apparel_df.to_pickle(filename)

    return apparel_df


def filter_primary_attributes(rag_kb, subCategory, articleType, baseColor,baseColorSimple):
    a = rag_kb[rag_kb['subCategory']==subCategory]
    b = a[a['articleType']==articleType]
    c= b[b['baseColour'].str.contains(baseColor, na=False) | b['baseColour'].str.contains(baseColorSimple, na=False)]
    c.reset_index(drop=True, inplace=True)
    return c


def generate_embedding(text, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name) 
    embedding = model.encode(text).tolist()
    return np.array(embedding).reshape(1,-1)


def get_nearest_neighbors(embeddings, query_embedding, n_neighbors=10):
    neigh = NearestNeighbors(n_neighbors=min(n_neighbors, len(embeddings)))
    neigh.fit(embeddings)
    idxs = neigh.kneighbors(query_embedding)[1][0]
    return idxs



PROMPT_FINAL_v1 = """
    You are a fashion expert. You are given an image of a model.
    
    You are furthermore given several clothing items, You are provided the product_id and the image of the item
    Sort these items by which most closely resembles the one worn by the model . Only judge similarity using the images provided. 
    
    Note that the first image you get is the image of the model . Subsequent images are the clothing items to compare against the model
    The subsequent images will be provided in the order of the  product ids you get below.
    
    
    
    Some Examples:
    - If the clothing item the model is wearing has sleeves, it is  disimilar to find an item with no sleeves
    - If the clothing the model is wearing is a formal shirt, it is  disimilar to find an item that is a t-shirt
    - If the clothing item the model is wearing has no design, it is disimilar to find an item with a very large colorful design
    
    
Return STRICT JSON (no prose) with this schema:
{
    "items": [
        {
        "product_id":"<product_id>",
        "similarity_order": "<order in similarity, 1 being most similar>"
        }
    ]
}

Rules:
- Keep fields short; no bounding boxes; no explanations.

The following is the product ids.


"""

def construct_selection_prompt(products_names, product_ids):
    products_list = ""
    for name, pid in zip(products_names, product_ids):
        products_list += f"- {name}, {pid}\n"
    return PROMPT_FINAL_v1 + "" + products_list


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def encode_image_url_to_base64(image_url):
    response = requests.get(image_url)
    img_bytes = response.content
    ## reduce image to 200x200

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    img_base_64 = resize_base64_image(img_base64, target_size=(200, 200))    
    return img_base_64

def send_selection_prompt(model_image_path , item_image_urls, prompt):
    client = OpenAI(api_key=API_KEY)


    model_image_base64 = encode_image_to_base64(model_image_path)

    image_base64_list = []
    for image_path in item_image_urls:
        image_base64 = encode_image_url_to_base64(image_path)
        image_base64_list.append(image_base64)


    response = client.chat.completions.create(
        model="gpt-5-nano",
        store= True,
        top_p=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{model_image_base64}"
                        }
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        } for img_base64 in image_base64_list
                    ]

                ],
            }
        ],
    )

    return response.choices[0].message.content


images_dict = read_images_csv()
# rag_kb = setup_knowledge_base_for_rag_v2("hf-hub:Marqo/marqo-ecommerce-embeddings-L")

rag_kb = setup_knowledge_base_for_rag("all-mpnet-base-v2")
items_str = get_llm_prediction_for_image()
items_json = json.loads(items_str)
items = items_json['items']

print("LLM predicted {x} items".format(x=len(items)))

for item in items:
    pprint(item)

    ## Filter out items based on subCategory, articleType, color_primary etc
    subCategory = item["category"]
    articleType = item["article_type"]
    baseColor = item["color_primary"]
    baseColorSimple = item["color_primary_simple"]
    
    filter_rag_kb = filter_primary_attributes(rag_kb, subCategory, articleType, baseColor,baseColorSimple)
    
    if filter_rag_kb.shape[0] == 0:
        print("No items found for:", item)
        continue
    
    print("Filtered RAG KB size:", filter_rag_kb.shape)
    
    
    query_embedding = generate_embedding(item['product_display_name'] + " " + " ".join(item["details"]), model_name='all-mpnet-base-v2')


    ids = get_nearest_neighbors(filter_rag_kb['embedding'].tolist(), query_embedding, n_neighbors=10)
    print("Top similar items from RAG KB based on KNN:")
    
    img_urls = []
    images = []
    product_display_names = []
    product_ids = []
    
    for id in ids:
        product_display_name= filter_rag_kb.iloc[id]['productDisplayName']
        product_id = filter_rag_kb.iloc[id]['id']
        product_ids.append(product_id)
        
        print(product_display_name, product_id)
        
        # Get image url and image
        img_url = images_dict[str(product_id)+".jpg"]
        print(img_url)
        response = requests.get(img_url)
        image_data = BytesIO(response.content)
        img = Image.open(image_data)
        
        # Store stuff 
        images.append(img)
        img_urls.append(img_url)
        product_display_names.append(filter_rag_kb.iloc[id]['productDisplayName'])
    # display images via matplot lib in one subplot
    for idx,img in enumerate(images):
        plt.subplot(1, len(img_urls), idx+1)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()   

    selection_prompt = construct_selection_prompt([""]*len(product_ids), product_ids)
    
    selection_response_str = send_selection_prompt(IMAGE_PATH , img_urls, selection_prompt)
    selection_response_json = json.loads(selection_response_str)
    
    print("Selection Response:")
    pprint(selection_response_json)
    
    items_selected = selection_response_json['items']
    items_with_order = sorted(items_selected, key=lambda x: int(x['similarity_order']))[:5]
    
    print("Top items after LLM selection:")
    selected_item_urls = []
    for selected_item in items_with_order:
        selected_item_id_str = selected_item['product_id']
        selected_item_id = int(selected_item_id_str)
        index_in_ids = product_ids.index(selected_item_id)
        print(product_display_names[index_in_ids], selected_item_id)
        selected_item_urls.append(img_urls[index_in_ids])
        
    for idx,img_url in enumerate(selected_item_urls):
        response = requests.get(img_url)

        image_data = BytesIO(response.content)
        img = Image.open(image_data)
        
        plt.subplot(1, len(selected_item_urls), idx+1)
        
        plt.imshow(img)


    plt.tight_layout()
    plt.show()    
        
    print("\n")


