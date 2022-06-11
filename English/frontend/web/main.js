function getFakeorTrue() {
    var data = document.getElementById('data').value
    eel.predict(data)(windowalert)
}

function windowalert(data) {
    alert(data)
}