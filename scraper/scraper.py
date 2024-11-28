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
        """Gets the rating of a post

        Returns:
            str: rating. One of 'general' or 'explicit'. But should always be 'general'. 
        """
        return self.rating
    
    def get_preview(self):
        """Gets the url of the preview image

        Returns:
            str: url of preview image 
        """
        return self.preview_url
    
    def get_sample(self):
        """Gets sample url of image

        Returns:
            str: sample url of image 
        """
        return self.sample_url
    
    def get_file(self):
        """Gets full sized image of post

        Returns:
            str: full sized image of post 
        """
        return self.file_url
    
    def get_height(self):
        """Height of image

        Returns:
            int: height of the image 
        """
        return self.height
    
    def get_width(self):
        """Gets width of image

        Returns:
            int: width of image
        """
        return self.width
    
    def get_pid(self):
        """Gets the page ID of the url to the JSON database. 
        Each unique page on Safebooru has a PID assigned to it,
        that increments each time the user goes to the next page.
        
        Note that the PID works differently for the JSON database,
        and the actual page itself. On the actual website, each page
        contains 42 images, and the PID is given by the index of the 
        first image on each page. So, on page one the PID will be 0,
        and on page two the PID will be 42. 
        
        However, the JSON database stores information on the first 100 
        posts in the database. Then, the next page can be accessed by changing
        the PID to be 2. So, on the JSON database, the PID database increments
        by 1 for each new page.

        Returns:
            pid: page ID 
        """
        return self.pid
    
    def set_pid(self, pid):
        """Changes the pid

        Args:
            pid (int): new page ID 
        """
        self.pid = pid
    
    def get_tag(self):
        """Tag associated to the image (e.g. fox_girl, blonde, etc.)

        Returns:
            str: tag associated to image 
        """
        return self.tag
    
    def set_tag(self, tag):
        """Changes the tag

        Args:
            tag (str): the new tag 
        """
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
            bool: True if image contains image, False otherwise 
        """
        mimetype, encoding = mimetypes.guess_type(url)
        return (mimetype and mimetype.startswith('image'))
    
    def _is_gif(self, url):
        """Checks whether the url contains a gif
        
        Returns:
            bool: True if image contains gif, False otherwise
        """
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
    
    def save_to_training(self, root_dir, tag, compressed=False):
        """Saves training images to root_dir

        Args:
            root_dir (str): the directory that the files need to be saved to
            tag (str): the tag (e.g. fox_girl) 
            compressed (bool, optional): whether or not to download the preview image (compressed),
            or to download the sample image (slightly compressed). There's the option
            to download the full-sized image as well, but that's not recommended. Defaults to 
            compressed
        """
        pid, i = 0, 0
        
        #  initialise first batch of posts
        posts = self._get_posts(pid, tag)
        
        while i < self.train_size:
            #  each page in the JSON database contains 100 datapoints
            #  PID needs to be increased after 100 points are read, which
            #  will take us to the next page of the database
            if i % 100 == 0 and i != 0:
                pid += 1 
                
                try:
                    posts = self._get_posts(pid, tag)
                #  this means that there are no more files to read
                except requests.exceptions.JSONDecodeError:
                    break
                
            dir = os.path.join(root_dir)
            
            for post in posts:
                #  get sample url, which is a slightly compressed version of the 
                #  full-sized image
                if compressed is False:
                    #  get sample url
                    url = post.get_sample()
                else:
                    #  get preview url
                    url = post.get_preview()
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
    