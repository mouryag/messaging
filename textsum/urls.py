from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from . import views
from .views import geeks_view, nav_view,first

urlpatterns = [
    url('^$',first,name='first'),
    url('^1/', geeks_view, name="template1"),
    path('2/', nav_view, name="template2"),
]