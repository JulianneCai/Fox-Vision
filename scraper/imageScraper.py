import mimetypes
import requests
import os

from io import BytesIO


class SafebooruPost:
    """ Class representing a Safebooru post for images of foxes
    
    Args:
        rating (str): one of 'general' or 'explicit'. Since it is from Safebooru, it should be 
        all rated 'general', but we should still check anyways.
        preview_url (str): url of the preview image
        sample_url (str): a url of a compressed version of the image
        file_url (str): a url of the full-sized image
        height (int): height of the image
        width (int): width of the image
    """
    def __init__(
        self, 
        rating, 
        preview_url, 
        sample_url, 
        file_url, 
        height, 
        width
    ):
        self.rating = rating
        self.preview_url = preview_url
        self.sample_url = sample_url
        self.file_url = file_url
        self.height = height
        self.width = width
    
    def get_rating(self):
        return self.rating
    
    def get_preview(self):
        return self.preview_url
    
    def get_sample(self):
        return self.sample_url
    
    def get_file(self):
        return self.file_url
    
    def get_height(self):
        return self.height
    
    def get_width(self):
        return self.width
    
    def _is_image(self, url):
        mimetype, encoding = mimetypes.guess_type(url)
        return (mimetype and mimetype.startswith('image'))
    
    def _is_gif(self, url):
        content_type = requests.head(url).headers['Content-Type']
        if content_type == 'image/gif':
            return True
        return False
    
class SafebooruScraper:
    def __init__(self, json_url):
        self.json_url = json_url
    
    def _get_posts(self):
        all_post_data = requests.get(self.json_url).json()
        
        posts = []
        
        for data in all_post_data:
            post = SafebooruPost(
                rating=data['rating'],
                preview_url=data['preview_url'],
                sample_url=data['sample_url'],
                file_url=data['file_url'],
                width=data['width'],
                height=data['height']
            )
            posts.append(post)
            
        return posts
    
    def save_to_training(self, root_dir):
        posts = self._get_posts()
        
        dir = os.path.exists(root_dir)
        for post in posts:
            url = post.get_sample()
            
            response = requests.get(url)
            if response.status_code == 200: 
                with open(root_dir, 'wb') as file:
                    file.write(response.content)
            else:
                print(f'Failed to download image. Status code: {response.status_code}.')
    
    
if __name__ == '__main__':
    json_url = 'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=fox_girl&json=1'
    
    scraper = SafebooruScraper(json_url)
    
    TRAINING_DIR = 'fox-data/train/fox-girl'
    
    scraper.save_to_training(TRAINING_DIR)
    
