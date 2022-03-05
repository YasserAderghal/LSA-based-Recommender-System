from django.shortcuts import render

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .utils import get_db_handle, get_collection_handle
from django.utils.functional import cached_property
from bson.json_util import dumps


# Create your views here.



db_handle, mongo_client = get_db_handle("movie", "localhost", 25000, None, None)

all_movies = list(db_handle["imdb"].find())



def home(request):
    page = request.GET.get('page', 1)
    

    num = 12


    paginator = Paginator(all_movies, num)

    try:
        movs = paginator.page(page)
    except PageNotAnInteger:
        movs = paginator.page(1)
    except EmptyPage:
        movs = paginator.page(paginator.num_pages)

    page_range = paginator.get_elided_page_range(number=page)

    context =  {'movies': movs, 'page_range':page_range } 
    return render(request , 'home/home.html', context)



def movie(request, pk):

    movie =  list(db_handle["imdb"].find({"movieId": pk })).pop()
    print(movie)
    context = {'id': pk , 'movie':movie}
    return render(request, 'home/movie.html', context)
