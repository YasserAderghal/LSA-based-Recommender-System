from django.shortcuts import render

# Create your views here.

rooms = [
    {'id': 1 , 'name' : 'text1'},
    {'id': 2 , 'name' : 'text2'},
    {'id': 3 , 'name' : 'text3'},
    {'id': 4 , 'name' : 'text4'},
]

def home(request):
        return render(request , 'home.html', {'rooms': rooms})



