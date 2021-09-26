let predRetrievedResult = {"msg":0};

const previewImage = () => {
    file = document.getElementById("query-image").files[0]
    if (file) {
        qimg.src = URL.createObjectURL(file)
        const reader = new FileReader();
        reader.onload = (e) => {
            const imgPath = e.target.result
            document.getElementById("query-url").value = imgPath
        }
        reader.readAsDataURL(file);
    }     
}

const previewImageURL= () => {
    qimg.src = document.getElementById("query-url").value;
}

const getStatusResult = (task_id) => {
    fetch(`/${task_id}`)
        .then(
            function(res){
                if (res.status == 200){
                    res.json().then((data) => {
                        predRetrievedResult = data
                    })
                    
                }
            } 
    ).catch(
        function(err){
            return {"status": 404, "msg":`Error ${err} occured. Please refresh page`}
            
        }
    )
}

const getPredResult = (task_id) => {
    fetch(`/result/${task_id}`)
        .then(
            function(res){
                if (res.status == 200){
                    res.json().then((data) => {
                        pimg.src = data["msg"]
                        document.getElementById("pred-caption").style.display = "block";
                        document.getElementById("pred-caption").innerHTML = "Right click to save image";
                    })
                    
                }
            } 
    ).catch(
        function(err){
            return {"status": 404, "msg":`Error ${err} occured. Please refresh page`}
            
        }
    )
}


const predImgDisp = (result) => {
   if (result["msg"] <= 50) {
      pimg.src  = "/static/img/img_process.png"
   }else {
      pimg.src  = "/static/img/still_process.jpeg"
   }
}

const task_id = document.getElementById("task_id").value;
let result = "";

if (task_id){
    let i = 0
    document.getElementById("query_status").innerHTML = "query submitted";
    const interval = 2000;
    let pString = "";
    let heartbeat = setInterval(
        () => { 
            console.log(`heartbeat ${i++}`);
            getStatusResult(task_id);
            document.getElementById("query_status").innerHTML = `${predRetrievedResult["msg"]}% Completed`
            predImgDisp(predRetrievedResult);

            if (predRetrievedResult["msg"] == 100 || i == 90){
                getPredResult(task_id)
                clearInterval(heartbeat);
            }
        }, interval);
}