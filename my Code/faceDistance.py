

knownHeight = 33 #CM
knownDistance = 135 #CM
pixel_height = 130 #pixelHeight

focalLength = (pixel_height * knownDistance) / knownHeight

def distance_to_camera(pixelHeight):
    # compute and return the distance from the image to the camera
    return (knownHeight * focalLength) / pixelHeight