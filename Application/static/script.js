fetch("/static/movies.csv")
.then(res => console.log(res))
.catch(err => console.log(err))