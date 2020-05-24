"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from . import views
from summarizer.views import two,enter

urlpatterns = [
    url(r'^',include('summarizer.urls')),
    url(r'two/',two,name='two'),
    url('admin/', admin.site.urls),
    url(r'^music/', include('music.urls')),
    url(r'^agri/', include('argiculture.urls')),
    url(r'new/',include('newone.urls')),
    url(r'chat/',include('chat.urls'))
]
