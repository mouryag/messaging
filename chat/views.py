from django.shortcuts import render,redirect
from .models import ChatBox,person

# Create your views here.

def home(request):
    all = []
    for i in ChatBox.objects.all():
        k=str(i.message)
        if k[0]=='0':
            print(k)
            all.append({0:k[1:]})
        else:
            all.append({1:k})
            print(k,"m")
    return render(request,'chat/chat.html',{'all':all})
def todatabase(request):
    if request.POST['mess']!='':
        a=ChatBox(message=request.POST['mess'])
        a.save()
    all=[]
    for i in ChatBox.objects.all():
        k=str(i.message)
        if k[0]=='0':
            print(k)
            all.append({0:k[1:]})
        else:
            all.append({1:k})
            print(k,"m")

    return redirect('home')
def enter(request):
    return render(request,'chat/name.html')
