from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns


urlpatterns = [
    path(r'',views.home,name='home'),
    #url(r'^music/', include('music.urls')),
]