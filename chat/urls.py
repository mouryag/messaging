from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from . import views

urlpatterns=[
    path('',views.home,name='home'),
    path('todatabase/',views.todatabase,name='todatabase'),
    path('enter/',views.enter,name='enter'),
]