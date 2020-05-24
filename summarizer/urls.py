from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from . import views

urlpatterns=[
    path('',views.enter,name='enter'),
    url(r'^two/',views.two,name='two'),
]