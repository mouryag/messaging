from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render,get_object_or_404
from .models import Album
import requests as re
import numpy as np

# def index(request):
#     all_albums=Album.objects.all()
#     html=''
#     for album in all_albums:
#         url='/music/'+str(album.id)+'/'
#         html+="<a href='"+url+"'>"+str(album.id)+ ":"+album.album_title+"</a>"+"<br><br><br>"
#         return HttpResponse(html)
# def index(request):
#     all_albums=Album.objects.all()
#     template=loader.get_template('music/index.html')  # By default it looks in templates directory
#     context={
#         "all_albums":all_albums
#     }
#     return HttpResponse(template.render(context,request))
def index(request):
    all_albums=Album.objects.all()
    context={
        "all_albums":all_albums
    }
    return render(request,'music/index.html',context)
# def details(request,album_id):
#     return HttpResponse("<h2>Details for album id :"+str(album_id)+"</h2>")
def details(request,album_id):
    try:
        album=Album.objects.get(pk=album_id)
    except Album.DoesNotExist:
        raise   Http404("Album does not exist.")
    # instead of above try except.....write album=get_object_or_404(Album,pk=album_id)
    return render(request,'music/details.html',{'album':album})
def requ(request):
    # #k=re.get("https://w3schools.com/")
    # #return HttpResponse(k.text)
    # image_url = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"
    #
    # # URL of the image to be downloaded is defined as image_url
    # r = re.get(image_url)  # create HTTP response object
    #
    # # send a HTTP request to the server and save
    # # the HTTP response in a response object called r
    # with open("python_logo.png", 'wb') as f:
    #     # Saving received content as a png file in
    #     # binary format
    #
    #     # write the contents of the response (r.content)
    #     # to a new file in binary mode.
    #     f.write(r.content)
    # return HttpResponse("<img src='python_logo.png'>")
    k=np.zeros(10)
    j=''
    for i in k:
        j+=str(i)
        j+='<br>'
    return HttpResponse(j)



