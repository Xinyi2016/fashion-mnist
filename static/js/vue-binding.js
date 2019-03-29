TWITTER_JSON = 'cm-55.json';
YOUTUBE_JSON = 'cm-3.json';

function parseResult(raw_str) {
    var lstBM = JSON.parse('[' + raw_str.trim().split('\n').join(',') + ']');
    $.each(lstBM, function (index) {
        $.each(Object.keys(lstBM[index]), function (idx, key) {
            lstBM[index]['_' + key] = lstBM[index][key]
        });
        lstBM[index]['_time_per_repeat'] = (lstBM[index]['done_time'] - lstBM[index]['start_time']) / lstBM[index]['num_repeat'];
        lstBM[index]['time_per_repeat'] = moment.duration(Math.ceil(lstBM[index]['_time_per_repeat']), "seconds").format("h:mm:ss", {trim: false});
        lstBM[index]['parameter'] = JSON.stringify(lstBM[index]['parameter']);
        lstBM[index]['mean_accuracy'] = lstBM[index]['mean_accuracy'].toFixed(3);
        lstBM[index]['std_accuracy'] = lstBM[index]['std_accuracy'].toFixed(3);
        lstBM[index]['start_time'] = moment(lstBM[index]['start_time'] * 1000).fromNow();
        lstBM[index]['done_time'] = moment(lstBM[index]['done_time'] * 1000).fromNow();
        lstBM[index]['processor'] = JSON.stringify(lstBM[index]['processor']);
        lstBM[index]['processor_para'] = JSON.stringify(lstBM[index]['processor_para']);
        lstBM[index]['topic_model'] = JSON.stringify(lstBM[index]['topic_model']);
        lstBM[index]['topic_para'] = JSON.stringify(lstBM[index]['topic_para']);
    });
    return lstBM
}

function loadResult(fn, cb) {
    $.ajax({
        type: "GET",
        url: fn,
        dataType: "text",
        success: function (data) {
            cb(parseResult(data));
        }
    });
}

const vm = new Vue({
    el: '#query2sku-ui',
    data: {
        bm_data: {
            'twitter': [],
            'youtube': [],
            'merge': []
        },
        sortKey: 'done_time',
        curDataName: 'twitter',
        search: '',
        sortOrder: -1,
        datasets: {
            'twitter': 'Twitter',
            'youtube': 'Youtube',
            'merge': 'Merged'
        },
        col_name_desc: {
            'name': 'Name',
            'parameter': 'Parameter',
            'processor': 'processor',
            'processor_para': 'processor_para',
            'topic_model': 'topic_model',
            'topic_para': 'topic_para',
            'mean_accuracy': 'Accuracy (mean)',
            'std_accuracy': 'Accuracy (std)',
            'time_per_repeat': 'Training time',
            'num_repeat': 'Repeats',
            'score': 'Score per repeat',
            'start_time': 'Job start',
            'done_time': 'Job Done',
            'm_mean_accuracy': 'Twitter Accuracy (mean)',
            'm_std_accuracy': 'Twitter Accuracy (std)',
            'z_mean_accuracy': 'YouTube Accuracy (mean)',
            'z_std_accuracy': 'YouTube Accuracy (std)'
        },
        col_show_name: {
            'twitter': ['name', 'parameter', 'processor', 'processor_para', 
            'topic_model', 'topic_para',
            'mean_accuracy', 'std_accuracy',
                'time_per_repeat', 'done_time'],
            'youtube': ['name', 'parameter', 'processor', 'processor_para', 
            'topic_model', 'topic_para',
            'mean_accuracy', 'std_accuracy',
                'time_per_repeat', 'done_time'],
            'merge': ['name', 'parameter', 'z_mean_accuracy', 'm_mean_accuracy', 'z_std_accuracy', 'm_std_accuracy']
        }
    },
    ready: function () {
        loadResult(TWITTER_JSON, function (data) {
            vm.bm_data['twitter'] = data;
            loadResult(YOUTUBE_JSON, function (data) {
                vm.bm_data['youtube'] = data;
                vm.bm_data['merge'] = vm.merge();
            });
        });
    },
    methods: {
        sortBy: function (sortKey) {
            this.sortOrder *= -1;
            this.sortKey = sortKey;
        },
        merge: function () {
            var tmp = {};
            var result = [];
            $.each(this.bm_data['youtube'], function (idx, data) {
                tmp[data['name'] + data['parameter']] = {
                    'm_mean_accuracy': data['mean_accuracy'],
                    'm_std_accuracy': data['std_accuracy'],
                    'z_mean_accuracy': 0,
                    'z_std_accuracy': 0,
                    'name': data['name'],
                    'parameter': data['parameter']
                };
            });
            $.each(this.bm_data['twitter'], function (idx, data) {
                if (data['name'] + data['parameter'] in tmp) {
                    tmp[data['name'] + data['parameter']]['z_mean_accuracy'] = data['mean_accuracy'];
                    tmp[data['name'] + data['parameter']]['z_std_accuracy'] = data['std_accuracy'];
                    tmp[data['name']] = data['name'];
                    result.push(tmp[data['name'] + data['parameter']])

                }
            });

            $.each(result, function (index) {
                $.each(Object.keys(result[index]), function (idx, key) {
                    result[index]['_' + key] = result[index][key]
                });
            });

            return result;
        }
    }
});