function getFakeorTrue() {
    var data = document.getElementById('data').value
    var model = document.getElementById('model').options[document.getElementById('model').selectedIndex].value 
    var embedding = document.getElementById('embeddings').options[document.getElementById('embeddings').selectedIndex].value 
    eel.predict(data, model, embedding)(windowalert)
}

function windowalert(data) {
    alert(data)
}