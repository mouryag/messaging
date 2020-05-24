from django.urls import path
from django.conf.urls import url,include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

app_name='music'   # NAMESPACE {% url 'music:detail' album.id %}
urlpatterns = [
    #url(r'^$', views.index,name='index'),
    # /music/71/
    url(r'^(?P<album_id>[0-9]+)/$',views.details,name='details'),
    #url(r'^(?P<album_id>[0-9]+)/favorite/$',views.favorite,name='favorite')
    url(r'^$',views.index,name='index')
]