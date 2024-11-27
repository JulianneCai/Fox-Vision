import mimetypes
import requests
import os


class SafebooruPost:
    """ Class representing a Safebooru post for images of foxes. 
    Given a tag and a pid, all other attributes can be derived from the 
    JSON data file.
    
    Args:
        pid (int): the page ID 
        tag (str): the tags associated to the image
        rating (str, optional): one of 'general' or 'explicit'. Since it is from Safebooru, it should be 
        all rated 'general', but we should still check anyways. Defaults to None.
        preview_url (str, optional): url of the preview image. Defaults to None.
        sample_url (str, optional): a url of a compressed version of the image. Defaults to None.
        file_url (str, optional): a url of the full-sized image. Defaults to None.
        height (int, optional): height of the image. Defaults to None.
        width (int, optional): width of the image. Defaults to None.
    """
    def __init__(
        self, 
        pid,
        tag,
        rating=None, 
        preview_url=None, 
        sample_url=None, 
        file_url=None, 
        height=None, 
        width=None,
    ):
        self.pid = pid
        self.tag = tag
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
    
    def get_pid(self):
        return self.pid
    
    def set_pid(self, pid):
        self.pid = pid
    
    def get_tag(self):
        return self.tag
    
    def set_tag(self, tag):
        self.tag = tag
        
    def get_json_url(self):
        """Constructs the link the the JSON database using the tag and the pid.

        Returns:
            str: url to the JSON database 
        """
        json_url = f'https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={self.tag}&pid={str(self.pid)}&json=1'
        return json_url
    
    def _is_image(self, url):
        """Checks whether the url contains an image

        Args:
            url (str): the url 

        Returns:
            bool: True if image contains url, False otherwise 
        """
        mimetype, encoding = mimetypes.guess_type(url)
        return (mimetype and mimetype.startswith('image'))
    
    def _is_gif(self, url):
        content_type = requests.head(url).headers['Content-Type']
        if content_type == 'image/gif':
            return True
        return False
    
    
class SafebooruScraper:
    """Scrapes Safebooru for images
    
    Args:
        train_size (int): number of images required for training set
    """
    def __init__(self, train_size):
        self.train_size = train_size
        
    def get_train_size(self):
        return self.train_size
    
    def set_train_size(self, train_size):
        self.train_size = train_size
    
    def _get_posts(self, pid, tag):
        """Given pid and tag, we access the JSON database, and use the
        datapoints to create a SafebooruPost object, and then store that 
        in a list.

        Args:
            pid (int): page id 
            tag (str): tags associated with desired image 

        Returns:
            List<SafebooruPost>: a list of SafebooruPosts from the JSON database
        """
        initial = SafebooruPost(pid=pid, tag=tag)
        
        json_url = initial.get_json_url()
        print(json_url)
        
        all_post_data = requests.get(json_url).json()
        
        posts = []
        
        for data in all_post_data:
            post = SafebooruPost(
                rating=data['rating'],
                preview_url=data['preview_url'],
                sample_url=data['sample_url'],
                file_url=data['file_url'],
                width=data['width'],
                height=data['height'],
                pid=pid,
                tag=tag
            )
            posts.append(post)
            
        return posts
    
    def save_to_training(self, root_dir, tag):
        """Saves training images to root_dir

        Args:
            root_dir (str): the directory that the files need to be saved to
            tag (str): the tag (e.g. fox_girl) 
        """
        pid, i = 0, 0
        
        while i < self.train_size:
            if i % 100 == 0 and i != 0:
                pid += 42
                try:
                    posts = self._get_posts(pid, tag)
                #  this means that there are no more files
                except requests.exceptions.JSONDecodeError:
                    break
            dir = os.path.join(root_dir)
            for post in posts:
                url = post.get_sample()
                response = requests.get(url, stream=True)
                if response.status_code == 200: 
                    print(i, 'Response granted. Writing...')
                    with open(os.path.join(dir, 'fox-girl-' + str(i) + '.jpg'), 'wb') as file:
                        file.write(response.content)
                        file.close()
                    print(i, 'File written')
                else:
                    print(f'Failed to download image. Status code: {response.status_code}.')
                i += 1
    
    
if __name__ == '__main__':
    scraper = SafebooruScraper(train_size=4000)
    
    TRAINING_DIR = 'D:/fox-detector/fox-data/train/fox-girl'
    
    scraper.save_to_training(root_dir=TRAINING_DIR, tag='fox_girl')
    