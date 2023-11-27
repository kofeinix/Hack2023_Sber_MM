from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.


def get_image(request):
    print("sdf")
    # image_data = byte_file.getvalue()
    return HttpResponse(request, content_type="image/jpeg")


