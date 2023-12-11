var fs = require('fs');
var path = require('path');
var readline = require('readline');
var stream = require('stream');
var zlib = require('zlib');
var Metrics = require('./Metrics.js');

function Evaluator(truthFile) {
  this.truth = {};       // id -> {synset, subsynset}
  this.bySynset = {};    // synset -> [id]
  this.bySubSynset = {}; // subsynset -> [id]
  this.bySplitSynset = {train: {}, val: {}, test: {}};  // split -> synset -> [id]
  this.truthFile = truthFile || 'all.csv.gz';
}
Evaluator.prototype.constructor = Evaluator;

Evaluator.prototype.init = function(callback) {
  var truthFile = this.truthFile;
  var scope = this;
  var processTruthFileLine = function(line) {
    var tokens = line.split(',');
    var id = tokens[0];
    if (id === 'id') { return; }  // this is the header line so skip it
    var synset = tokens[1];
    var subsynset = tokens[2];
    var split = tokens[4];
    scope.truth[id] = {synset: synset, subsynset: subsynset, split: split};

    if (!scope.bySynset[synset]) scope.bySynset[synset] = [];
    scope.bySynset[synset].push(id);

    if (!scope.bySubSynset[subsynset]) scope.bySubSynset[subsynset] = [];
    scope.bySubSynset[subsynset].push(id);

    var splitBin = scope.bySplitSynset[split];
    if (!splitBin[synset]) splitBin[synset] = [];
    splitBin[synset].push(id);
  };

  if (truthFile.endsWith('.gz')) {
    readline.createInterface({
      input: fs.createReadStream(truthFile).pipe(zlib.createGunzip())
    }).on('line', function(line) {
      processTruthFileLine(line);
    }).on('close', function() {
      console.log('Loaded Evaluator using file ' + truthFile);
      callback();
    });
  } else {
    fs.readFileSync(truthFile).toString().split('\n').forEach(function(line) {
      processTruthFileLine(line);
    });
    console.log('Loaded Evaluator using file ' + truthFile);
    callback();
  }
};

Evaluator.prototype.resultsToScores = function(queryId, results) {
  var query = this.truth[queryId];
  var scores = [];
  for (var i = 0; i < results.length; i++) {
    var result = this.truth[results[i]];
    if (result === undefined) {
      // retrieved result is unknown so ignore it
      //console.error("retrieved id="+results[i]+" is unknown for query id="+queryId);
      continue;
    }
    if (result.split !== query.split) {
      // retrieved result is coming from a different split so ignore it
      //console.error("retrieved id="+results[i]+" is in split different from query id="+queryId);
      continue;
    }
    if (query.synset !== result.synset) {  // parent synset mismatch
      scores.push(0);
    } else {  // parent synset match, check subsynset
      if (query.subsynset === result.subsynset) {  // subsynset match
        scores.push(3);
      } else if (query.synset === result.subsynset) {  // predicted subsynset is query synset
        scores.push(2);
      } else {  // must be a sibling subsynset since match
        scores.push(1);
      }
    }
  }
  //console.log(queryId + ":" + scores);
  return scores;
};

// Pad number n to width p by adding zeroes on left
var zeroPad = function(n, p) {
  var pad = new Array(1 + p).join('0');
  return (pad + n).slice(-pad.length);
};

// Reads all query results from given dir
var readQueryResults = function(dir) {
  var files = fs.readdirSync(dir);
  var allQueryResults = {};

  // helper converts line in results file and pushes to results
  var lineToQueryResult = function(results, line) {
    if (line) {
      var tokens = line.split(' ');
      var id = tokens[0];
      if (id.length !== 6) { id = zeroPad(id, 6); }  // handle non-padded ids
      //var dist = tokens[1];  // ignore dists for now
      results.push(id);
    }
  };

  // iterate over files and convert each to results list
  for (var iF = 0; iF < files.length; iF++) {
    var file = files[iF];
    var results = [];
    var line2result = lineToQueryResult.bind(undefined, results);
    fs.readFileSync(dir + file).toString().split('\n').forEach(line2result);
    var id = (file.length === 6) ? file : zeroPad(file, 6);  // handle non-padded ids
    allQueryResults[id] = results;
  }

  return allQueryResults;
};

// save summary metrics for each id in avgs
var saveEvaluationStats = function(results, method, allCategoryPRs) {
  // summary metrics output file
  var outSum = fs.createWriteStream(method + '.summary.csv');
  outSum.on('error', function(err) { console.error(err); });
  outSum.write('class,P@N,R@N,F1@N,mAP,NDCG,dataset,method\n');

  // PR value output file
  var outPR = fs.createWriteStream(method + '.pr.csv');
  outPR.on('error', function(err) { console.error(err); });
  outPR.write('class,P,R,dataset,method\n');

  for (var set in results) {
    if (!results.hasOwnProperty(set)) { continue; }
    var avgs = results[set];
    for (var id in avgs) {
      if (!avgs.hasOwnProperty(id)) { continue; }
      var a = avgs[id].getAverages();

      // write summary metrics row
      var l = [id, a['P@N'], a['R@N'], a['F1@N'], a.mAP, a.NDCG, set, method].join(',');
      console.log(l);  outSum.write(l + '\n');

      // skip category PR metrics unless allCategoryPRs specified
      if (id === 'microALL' || id === 'macroALL' || allCategoryPRs) {
        for (var k = 0; k < a.P.length; k++) {
          var pr = [id, a.P[k], a.R[k], set, method].join(',') + '\n';
          outPR.write(pr);
        }
      }
    }
  }

  outSum.end();
  outPR.end();
};

// Evaluates set of ranked lists contained in dir returning set of summary metrics
Evaluator.prototype.evaluateRankedLists = function(dir) {
  var dirTokens = dir.replace(/\\/g, '/').split('/').filter(function(s) { return s.length > 0; });
  var split = dirTokens.pop().split('_')[0];
  var queries = readQueryResults(dir);
  var cutoff = 1000;  // only consider recall up to this retrieval list length
  var metrics = {'microALL': new Metrics.SummedMetrics()};
  for (var queryId in queries) {
    if (!queries.hasOwnProperty(queryId)) { continue; }
    var queryTruth = this.truth[queryId];
    if (queryTruth === undefined) {
      // Query model is unknown, so just ignore this query result
      continue;
    }
    if (queryTruth.split !== split) {
      console.error('Ignoring query model from ' + queryTruth.split + ' found in ' + split);
      continue;
    }

    var querySynset = queryTruth.synset;
    var results = queries[queryId];
    if (results.length > cutoff) {  // only accept up to cutoff results
      results = results.slice(0, cutoff);
    }
    // filter out any duplicate retrieval ids
    var resultsSet = new Set();
    var filteredResults = [];
    for (var i = 0; i < results.length; i++) {
      if (!resultsSet.has(results[i])) {
        resultsSet.add(results[i]);
        filteredResults.push(results[i]);
      }
    }
    results = filteredResults;
    var oracleResults = this.bySplitSynset[split][querySynset];
    var scores = this.resultsToScores(queryId, results);
    var oracleScores = this.resultsToScores(queryId, oracleResults);
    metrics.microALL.addResult(scores, oracleScores, cutoff);
    if (!metrics[querySynset]) { metrics[querySynset] = new Metrics.SummedMetrics(); }
    metrics[querySynset].addResult(scores, oracleScores, cutoff);
  }

  // Computer macro average = average of per-synset micro averages
  metrics.macroALL = new Metrics.SummedMetrics();
  for (var synId in metrics) {
    if (metrics.hasOwnProperty(synId) && synId !== 'microALL') {
      metrics.macroALL.addSummedMetrics(metrics[synId]);
    }
  }

  return metrics;
};

// Evaluates all available train/val/test x normal/perturbed datasets in given
// dir and returns object containing {datasetName: result} key-values
Evaluator.prototype.evaluate = function(dir) {
  var datasetNames = [
  'test_normal',  'test_perturbed',
  'val_normal',   'val_perturbed',
  'train_normal', 'train_perturbed'
  ];

  // compute results for each dataset
  var results = {};
  for (var i = 0; i < datasetNames.length; i++) {
    var datasetName = datasetNames[i];
    var datasetDir = dir + datasetName + '/';
    try {
      var stats = fs.statSync(datasetDir);
      if (stats.isDirectory()) {
        console.log(datasetDir + '...');
        results[datasetName] = this.evaluateRankedLists(datasetDir);
      }
    } catch (err) {
      //console.error(err);
      continue;
    }
  }

  // save evaluation stats files
  var method = path.basename(dir);
  saveEvaluationStats(results, method);
  return results;
};

// Writes oracle query results to given dir, cutting off lists at maxN
Evaluator.prototype.writeOracleResults = function(dir, maxN) {
  // for each id in truth table
  for (var modelId in this.truth) {
    if (!this.truth.hasOwnProperty(modelId) || modelId === '') { continue; }
    var model = this.truth[modelId];
    var synsetId = model.synset;
    // get all ids with same synset
    var sameSynsetModelIds = this.bySynset[synsetId];
    if (sameSynsetModelIds.length > maxN) {
      sameSynsetModelIds = sameSynsetModelIds.slice(0, maxN);
    }
    var file = dir + '/' + modelId;
    fs.writeFileSync(file, sameSynsetModelIds.join('\n'));
  }
};

module.exports = Evaluator;
