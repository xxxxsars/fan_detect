/**
 * Created by Andy on 2021/2/24.
 */

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}




function take_snapshot(element) {

    let img = ""
    // take snapshot and get image data
    Webcam.snap(function (data_uri) {
        element.html('<img  src="' + data_uri + '"/>');
        img = data_uri
    });

    return img
}

function setup(element) {
    Webcam.reset();
    Webcam.attach(element);
}


