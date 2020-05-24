from django.shortcuts import render


# create a function
def first(request):
    return render(request,"textsum/template1.html")
def geeks_view(request):
    # return response
    return render(request, "textsum/geeks.html")


def nav_view(request):
    # return response
    return render(request, "textsum/nav.html")