ImagePatchPrompt = f'''
import math
from PIL import Image
class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left, lower, right, upper : int
        An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.
        
    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.
        Parameters
        -------
        image : PIL.image
        left, lower, right, upper : int
            An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.
        """
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    def compute_depth(self):
        """Returns the median depth(the distance of object to camera) of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop

        Examples
        --------
        >>> # Query:What is front,apple or banana?
        >>> # Usable_image_patches:apple_patch,banana_patch;
        >>> # The variable you should return: answer:'apple' or 'banana'
        >>> def execute_command(apple_patch,banana_patch):
        >>>     apple_distance = apple_patch.compute_depth()
        >>>     banana_distance = banana_patch.compute_depth()
        >>>     if apple_distance < banana_distance:
        >>>         return apple
        >>>     else:
        >>>         return banana

        """
        depth_map = compute_depth(self.cropped_image)
        return depth_map.median()
'''
_finder_api_prompt_v3 = '''
    # Suitable for situations where specific objects need to be found
    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        list[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # Query:find the foo;
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: foo_patches:foo in the image
        >>> def execute_command(image_patch) -> list[ImagePatch]:
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches


        >>> # Query:find the runway and the staircase in the image 
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: runway_patch: The patch of the runway in the image, staircase_patch: The patch of the staircase in the image
        >>> def execute_command(image_patch) -> list[ImagePatch]:
        >>>    runway_patches = image_patch.find("runway")
        >>>    staircase_patches = image_patch.find("staircase")
        >>>    return runway_patches,staircase_patches

        
        >>> # Query:find the second bar;
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: bar_patch:The second bar in this image
        >>> def execute_command(image_patch) -> str:
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch

        >>> # Query:find the black foo on top of the qux;
        >>> # Usable_image_patches:image_patch; 
        >>> # The variable you should return: foo_patch:The balck foo which is on top of the qux
        >>> def execute_command(image_patch) -> ImagePatch:
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             return foo

        
        >>> # Query:find the object on which the seat is located ;
        >>> # Usable_image_patches:image_patch,seat_patch; 
        >>> # The variable you should return: object_under_seat: The object on which the seat is located
        >>> def execute_command(seat_patch) -> ImagePatch:
        >>>     object_patches = seat_patch.find("seat")
        >>>     for obj in object_patches:
        >>>         if obj.vertical_center < seat_patch.vertical_center:
        >>>             return obj

    # Suitable for situations where the name of object is unknown and content in a certain direction in the picture needs to be located.
    def locate(self,direction:Literal['right','left','up','bottom']):
        """Return an ImagePatch contains everything in the specified direction of current ImagePatch.
        Parameters
        ----------
        direction : Literal['right','left','up','bottom']
            The direction need to reamin.
        >>> # Query:find the object in the lower right corner of the image;
        >>> # Input_patches:image_patch;
        >>> # The variable you should return: object_low_right_patch:The object in the low right corner of the image
        >>> def execute_command(image_patch) -> ImagePatch:
        >>>     right_patch = image_patch.locate('right')
        >>>     object_low_right_patch = image_patch.locate('bottom')
        >>>     return object_low_right_patch

        >>> # Query:find the foo in the left of the aux;
        >>> # Input_patches:aux_patch;
        >>> # The variable you should return: foo_patch:The foo in the image.
        >>> def execute_command(aux_patch) -> ImagePatch:
        >>>     left_patch = image_patch.locate('left')
        >>>     foo_patch = left_patch.find('foo')
        >>>     return foo_patch
        """
        return locate_in_image(self.cropped_image, direction)
'''
_finder_api_prompt = '''
    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        list[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # Query:find the foo;
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: foo_patches:foo in the image
        >>> def execute_command(image_patch) -> list[ImagePatch]:
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches


        >>> # Query:find the runway and the staircase in the image 
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: runway_patch: The patch of the runway in the image, staircase_patch: The patch of the staircase in the image
        >>> def execute_command(image_patch) -> list[ImagePatch]:
        >>>    runway_patches = image_patch.find("runway")
        >>>    staircase_patches = image_patch.find("staircase")
        >>>    return runway_patches,staircase_patches

        
        >>> # Query:find the second bar;
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: bar_patch:The second bar in this image
        >>> def execute_command(image_patch) -> str:
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch

        >>> # Query:find the black foo on top of the qux;
        >>> # Usable_image_patches:image_patch; 
        >>> # The variable you should return: foo_patch:The balck foo which is on top of the qux
        >>> def execute_command(image_patch) -> ImagePatch:
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             return foo

        
        >>> # Query:find the object on which the seat is located ;
        >>> # Usable_image_patches:image_patch,seat_patch; 
        >>> # The variable you should return: object_under_seat: The object on which the seat is located
        >>> def execute_command(seat_patch) -> ImagePatch:
        >>>     object_patches = seat_patch.find("seat")
        >>>     for obj in object_patches:
        >>>         if obj.vertical_center < seat_patch.vertical_center:
        >>>             return obj


        """
        return find_in_image(self.cropped_image, object_name)
'''

_finder_api_prompt_for_ram = '''
    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        list[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # Query:find the foo;
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: foo_patches:foo in the image
        >>> def execute_command(image_patch) -> list[ImagePatch]:
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches

        >>> # Query:find the runway and the staircase in the image 
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: runway_patch: The patch of the runway in the image, staircase_patch: The patch of the staircase in the image
        >>> def execute_command(image_patch) -> list[ImagePatch]:
        >>>    runway_patches = image_patch.find("runway")
        >>>    staircase_patches = image_patch.find("staircase")
        >>>    return runway_patches,staircase_patches

        >>> # Query:find the second bar;
        >>> # Usable_image_patches:image_patch;
        >>> # The variable you should return: bar_patch:The second bar in this image
        >>> def execute_command(image_patch) -> str:
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch

        >>> # Query:find the black foo on top of the qux;
        >>> # Usable_image_patches:image_patch; 
        >>> # The variable you should return: foo_patch:The balck foo which is on top of the qux
        >>> def execute_command(image_patch) -> ImagePatch:
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             return foo

        >>> # Query:find the object on which the seat is located ;
        >>> # Usable_image_patches:image_patch,seat_patch; 
        >>> # The variable you should return: object_under_seat: The object on which the seat is located
        >>> def execute_command(seat_patch) -> ImagePatch:
        >>>     object_patches = seat_patch.find("seat")
        >>>     for obj in object_patches:
        >>>         if obj.vertical_center < seat_patch.vertical_center:
        >>>             return obj

        """
        return find_in_image(self.cropped_image, object_name)
'''

_finder_api_prompt_V2 = '''
    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        list[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # Query: find the all foo; Usable_image_patches: image_patch; Return varname: foo_patches
        >>>     foo_patches = image_patch.find("foo")

        >>> # Query: find the second bar; Usable_image_patches: image_patch; Return varname: bar_patch
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]

        >>> # Query: find the black foo on top of the qux; Usable_image_patches:image_patch; Return varname: foo_patch
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             foo_patch = foo

        """
        return find_in_image(self.cropped_image, object_name)
'''

FinderImagePatchPrompt = f'''
    {ImagePatchPrompt}
    {_finder_api_prompt}
'''

FinderImagePatchPromptWithLocate = f'''
    {ImagePatchPrompt}
    {_finder_api_prompt_v3}
'''

_verifier_api_prompt = f'''
    def verify_property(self, object_name: str, visual_property: str) -> bool:
        """Returns True if the object possesses the visual property, and False otherwise.
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        visual_property : str
            A string describing the simple visual property (e.g., color, shape, material) to be checked.

        Examples
        -------
        >>> # Query:Do the letters have blue color?;
        >>> # Usable_image_patches:image_patch,letters_patch; 
        >>> # The variable you should return: letters_blue_or_not:A bool value indicate whether the letter is blue or not
        >>> def execute_command(letters_patch) -> str:
        >>>     letters_blue_or_not = letters_patches.verify_property("letters", "blue")
        >>>     return letters_blue_or_not

        """
        return verify_property(self.cropped_image, object_name, property)

    def get_property(self, object_name: str, visual_property_class: str) -> str:
        """Returns visual property of the object in the image patch
        Differs from 'verify_property' in that it check whether the object possesses the specified property., instead of getting it from image.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        visual_property_class: str
            A simple visual property class(e.g., color, shape, material) to be get.

        Examples
        -------
        >>> # Query:Get the color of the helmet;
        >>> # Usable_image_patches:helmet_patch; 
        >>> # The variable you should return: color: The color of the helmet
        >>> # def execute_command(helmet_patch) -> None:
        >>>     color = helmet_patch.get_property("helmet", "color")
        >>>     return color

        """
        return get_property(self.cropped_image, object_name, property)
        
    def best_text_match(self, option_list: list[str]) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options

        Examples
        -------
        >>> # Query:Is the foo gold or white?;
        >>> # Usable_image_patches:image_patch,foo_patch; 
        >>> # The variable you should return: gold_or_white: The color of the foo.gold or white.
        >>> def execute_command(foo_patch)->str:
        >>>     gold_or_white = foo_patch.best_text_match(["gold", "white"])
        >>>     return gold_or_white 


        """
        return best_text_match(self.cropped_image, option_list)
'''

_verifier_api_prompt_V2 = f'''
    def verify_property(self, object_name: str, visual_property: str) -> bool:
        """Returns True if the object possesses the visual property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        visual_property : str
            A string describing the simple visual property (e.g., color, shape, material) to be checked.

        Examples
        -------
        >>> # Query: Do the letters have blue color?; Usable_image_patches: image_patch,letters_patch； Return varname: letters_blue_or_not
        >>>     letters_blue_or_not = letters_patches.verify_property("letters", "blue")
        """
        return verify_property(self.cropped_image, object_name, property)

    def best_text_match(self, option_list: list[str]) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options

        Examples
        -------
        >>> # Query:Is the foo gold or white?; Usable_image_patches:image_patch,foo_patch;  Return varname: gold_or_white
        >>>     gold_or_white = foo_patch.best_text_match(["gold", "white"])
        """
        return best_text_match(self.cropped_image, option_list)
'''

VerifierImagePatchPrompt = f'''
    {ImagePatchPrompt}
    {_verifier_api_prompt}
'''
_querier_api_prompt = f'''
    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------

        >>> # Query:Which kind of baz is not fredding?;
        >>> # Usable_image_patches:image_patch,baz_patches;
        >>> # The variable you should return:baz_type:The type of the baz
        >>> def execute_command(baz_patches) -> str:
        >>>     for baz_patch in baz_patches:
        >>>         baz_type = baz_patch.simple_query("What is this baz?")
        >>>         return baz_type

        >>> # Query:What color is the foo?;
        >>> # Usable_image_patches:image_patch,foo_patch; 
        >>> # The variable you should return:foo_color:The color of the foo
        >>> def execute_command(foo_patch) -> str:
        >>>     foo_color = foo_patch.simple_query("What is the color?")
        >>>     return foo_color


        """
        return simple_query(self.cropped_image, question)
'''
_querier_api_prompt_V2 = f'''
    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------

        >>> # Query: Which kind of baz is not fredding?; Usable_image_patches: image_patch,baz_patches; Return varname:baz_type
        >>>     for baz_patch in baz_patches:
        >>>         baz_type = baz_patch.simple_query("What is this baz?")

        >>> # Query: What color is the foo?; Usable_image_patches:image_patch,foo_patch;  Return varname:foo_color
        >>>     foo_color = foo_patch.simple_query("What is the color?")

        """
        return simple_query(self.cropped_image, question)
'''

QuerierImagePatchPrompt=f'''
    {ImagePatchPrompt}
    {_querier_api_prompt}
'''