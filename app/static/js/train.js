
function start_train() {
    if(training) return;

    if (typeof (dataset) === undefined) { // cus we add '' around jinja variable
        custom_alert('You mush choose a Dataset first!');
        return;
    }

    let epoch = $("#epoch_num").val();
    let start_epoch = $("#start_epoch").val();
    if(epoch === '' || epoch === undefined || epoch.indexOf(" ") >= 0 || isNaN(epoch) || epoch.indexOf('.') >= 0){
        custom_alert("Please input leagal epoch num");
        return;
    }

    if(start_epoch === '' || start_epoch === undefined || start_epoch.indexOf(" ") >= 0 || isNaN(start_epoch) || start_epoch.indexOf('.') >= 0){
        custom_alert("Please input leagal start epoch");
        return;
    }

    epoch = parseInt(epoch);
    start_epoch = parseInt(start_epoch);
    if (epoch <= 0){
        custom_alert("epoch out of bound");
        return
    }

    if (start_epoch < 0){
        custom_alert("start epoch out of bound");
        return
    }

    console.log("dataset is:", dataset);

    console.log('start training..');
    let model = $('#model_picker option:selected').text();
    let tricks = [];
    let pretrain = $('#pretrain_picker option:selected').text();

    $('#trick_picker option:selected').map(function () {
        tricks.push($(this).text());
    });
    let model_params = {
        'model': model,
        'dataset': dataset,
        'pretrain' : pretrain,
        'tricks': tricks.toString(), // cvt list to str
        'epoch': epoch,
        'start_epoch': start_epoch,
    };

    console.log(model_params);
    training = true;
    start_supervise();

    $.ajax({
        type: "post",
        url: "start_train",
        data: model_params,
        async: true,
        success: function (data) {
            console.log(data);
        }
    });
}

function update(data) {
    // $('#progress_bar').attr('aria-valuenow', data['progress']).css('width', data['progress'] + '%');
    // $('#progress_val').text(data['progress'] + '%');
    console.log('update status');
    console.log(data);
    cur_idx = parseInt(data['cur_idx']);
    trained_num = parseInt(data['trained']);

    $('#trained_num').text(trained_num);
    $('#total_num').text(data['total_num']);
    $('#training_idx').text(cur_idx);
    $('#training_stage').text(data['stage']);

    if (trained_num === 0) {
        ratio = 0;
    } else {
        ratio = trained_num / parseInt(data['total_num']) * 100;
        ratio = ratio.toFixed(2)
    }

    $('#progress_bar').attr('aria-valuenow', data['progress']).css('width', ratio + '%');
    $('#progress_val').text(ratio + '%');

    let msgs = data['msgs'];
    show_evals(msgs)

    change_control_status(data,training)
}

function show_evals(msgs) {
    let tab = document.getElementById("eval_body");
    let head = document.getElementById("eval_head");
    tab.innerHTML = "";
    head.innerHTML = "";
    for (let k in msgs) {
        let th = document.createElement("th");
        let td = document.createElement("td");
        th.innerHTML = k;
        td.innerHTML = msgs[k];

        head.appendChild(th);
        tab.appendChild(td);

    }
}

function stop_train() {
    console.log('stop training..');
    $.ajax({
        url: "stop_train",
        success: function () {
            custom_alert('stopping, please wait for the completion of this stage');
        }
    });
}

function visualize() {
    if ($("#visual_frame").is(":visible")) {
        $("#visual_frame").hide();
        return;
    }


    console.log('viz');
    $.ajax({
        url: "visualize_train",
        success: function (data) {
            if (data != '') {
                console.log(data)
                if(data == 'waiting'){
                    custom_alert("the tensorboard is starting, please wait")
                    return
                }else{
                    $("#visual_frame").attr("src", data[0])
                    $("#visual_frame").show()
                }
            } else {
                console.log('null address')
                $("#visual_frame").hide()
                custom_alert("there is no tebsorboard to be shown")
            }
        }
    });
}

function save_model(){
    var model = $('#model_picker option:selected').text();
    var msg = ("Are you sure you want to overwrite the best model?\n\nBe sure you know what you are doing!\n\nOverwrite the "
        + model + " on " + dataset);
    if (confirm(msg) == true){
        params = {
            'model': model,
            'dataset': dataset
        };
        $.ajax({
        url: "/train/replace_pretrain",
        data: params,
        success: function (data) {
            if(data == "OK"){
                custom_alert("replace success, now you can refresh the page to choose the new pretrain!")
            }else{
                custom_alert("something error when replacing, please check!")
            }
        }
    });
    }
}

function start_supervise() {
    training = true;
    let update_progress = setInterval(function () {
        $.ajax({
            type: "post",
            url: "train_status",
            async: true,
            success: function (data) {
                update(data);
                if (data['training'] === false) {
                    training = false;
                    clearInterval(update_progress);
                    change_control_status(data,training);
                    custom_alert("stopped!")
                }
            }
        })
    }, 800);
}

function custom_alert(msg) {
    let alerts = document.getElementById("alerts");
    alerts.innerHTML = "";
    let div = document.createElement("div");

    div.classList.add("alert");
    div.classList.add("alert-warning");
    let a = document.createElement("a");
    a.classList.add('close');
    a.setAttribute("href","#");
    a.setAttribute("data-dismiss","alert");
    a.innerHTML = "&times;";
    let strong = document.createElement("strong");
    strong.innerHTML = msg;
    div.appendChild(a);
    div.appendChild(strong);
    alerts.appendChild(div);
}

function change_control_status(status,training){
    if(model_tricks.hasOwnProperty(status['model'])){
        $("#model_picker").selectpicker('val', status['model']);
        $("#trick_picker").selectpicker('val', status['tricks']);
        $("#pretrain_picker").selectpicker('val', status['pretrain'])
    }

    if(training){
        $("#model_picker").prop("disabled",true);
        $("#trick_picker").prop("disabled",true);
        $("#pretrain_picker").prop("disabled",true);
        $("#model_picker").selectpicker('refresh');
        $("#trick_picker").selectpicker('refresh');
        $("#pretrain_picker").selectpicker('refresh');
    }else{
        $("#model_picker").prop("disabled",false);
        $("#trick_picker").prop("disabled",false);
        $("#pretrain_picker").prop("disabled",false);
        $("#model_picker").selectpicker('refresh');
        $("#trick_picker").selectpicker('refresh');
        $("#pretrain_picker").selectpicker('refresh');
    }

    $("#epoch_num").attr("disabled",training);
    $("#epoch_num").attr("value",status['epoch']);
    $("#start_epoch").attr("disabled",training);
    $("#start_epoch").attr("value",status['start_epoch']);
    $("#start_train_btn").attr('disabled', training);
    $("#stop_train_btn").attr('disabled', !training);

}

function set_listeners(){
    // change tricks according to chosen model
    $('#model_picker').on('changed.bs.select', function () {
        let options = model_tricks[$("#model_picker").val()]; // get tricks of select model
        $("#trick_picker option").remove(); // remove original
        options.map(function (op) {
            $('#trick_picker').append("<option value='" + op + "'>" + op + "</option>");
        });
        $('#trick_picker').selectpicker('refresh');

        let pretrain_options = pretrains[$("#model_picker").val()]; // get tricks of select model
        $("#pretrain_picker option").remove(); // remove original
        $('#pretrain_picker').append("<option value='None'>None</option>");
        if(pretrain_options) {
            pretrain_options.map(function (op) {
                $('#pretrain_picker').append("<option value='" + op + "'>" + op + "</option>");
            });
        }
        $('#pretrain_picker').selectpicker('refresh');

    });
}

function init(){
    if (training) {
        console.log("training");
        start_supervise();
    }
    $.ajax({
            type: "post",
            url: "train_status",
            async: true,
            success: function (data) {
                set_listeners();
                update(data);
                change_control_status(data,training);

            }
    });
}

init();
