var fs = require('fs');
var request = require('request');

function Taxonomy(params) {
  this.synsetMinNumInstances  = 10;
  this.models                 = {};         // {id, synsets, topLevelSynsetId}
  this.synsets                = {};         // taxonomy of synsets
  this.topLevelSynsets        = new Set();  // top-level parent synset ids
  this.topLevelSynsetModels   = {};         // topLevelSynsetId -> modelIds

  this.buildTaxonomy(params.taxonomyFile);
  this.populateModels(params.topLevelSynsetModelsFile);
}
Taxonomy.prototype.constructor = Taxonomy;

var getAndLoad = function(url, localFile, callback) {
  request(url).pipe(fs.createWriteStream(localFile));
  callback(fs.readFileSync(localFile, 'utf8'));
};

var getTaxonomyJSON = function(callback) {
  var url = "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/taxonomy.json";
  var file = 'tmp/taxonomy.json';
  getAndLoad(url, file, callback);
};

var getModelSynsets = function(callback) {
  var url = "http://shapenet.cs.stanford.edu/models3d/solr/select?q=datasets%3AShapeNetCore&start=0&rows=100000&fl=id%2Cwnsynset&wt=json";
  var file = 'tmp/model-synsets.json';
  getAndLoad(url, file, callback);
};

var getModelList = function(callback) {
  var url = "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/model-list.txt"
  var file = 'tmp/model-list.txt';
  getAndLoad(url, file, callback);
};

Taxonomy.prototype.buildTaxonomy = function(rawTax) {
  // copy synsets
  for (var i = 0; i < rawTax.length; i++) {
    var synset = rawTax[i];
    var synId = synset.synsetId;
    this.synsets[synId] = synset;
  }

  // populate parent pointers
  for (var synsetId in this.synsets) {
    var parent = this.synsets[synsetId];
    for (var iC = 0; iC < parent.children.length; iC++) {
      var childId = parent.children[iC];
      this.synsets[childId].parent = synsetId;
    }
  }
};

Taxonomy.prototype.populateModels = function(topLevelSynsetModelsFile) {
  var scope = this;

  // parse model synsets and create entries
  var tmp = JSON.parse(fs.readFileSync(modelSynsetsFile, 'utf8'));
  tmp.response.docs.map(function(e) {
    var id = e.fullId.split('.')[1];
    scope.models[id] = { id: id, synsets: e.wnsynset };
  });

  // parse toplevel model synset mappings and populate
  fs.readFileSync(topLevelSynsetModelsFile).toString().split('\n').map(function(s) {
    var tokens = s.split(',');
    var modelId = tokens[1];
    var topSynId = tokens[0];

    // any unknown models just belong to topSynId
    if (scope.models[modelId] === undefined) {
      scope.models[modelId] = {id: modelId, synsets: [topSynId]};
    }

    if (scope.topLevelSynsetModels[topSynId] === undefined) {
      scope.topLevelSynsetModels[topSynId] = [];
    }
    scope.topLevelSynsetModels[topSynId].push(modelId);
    scope.models[modelId].topLevelSynsetId = topSynId;

    scope.topLevelSynsets.add(topSynId);
  });
  //console.log(scope.models);
};

var getPathToTop = function(synsets, childId) {
  var path = [childId];
  var currParentSynsetId = synsets[childId].parent;
  while (currParentSynsetId) {
    path.unshift(currParentSynsetId);
    var currParent = synsets[currParentSynsetId];
    currParentSynsetId = currParent.parent;
  }
  return path;
};

Taxonomy.prototype.findSubSynsetForModel = function(modelId) {
  var model = this.models[modelId];
  var subSynsetCandidates = [];
  for (var i = 0; i < model.synsets.length; i++) {
    var synId = model.synsets[i];
    if (this.synsets[synId]) {
      var path = getPathToTop(this.synsets, synId);
      //if (modelId.substr(0,4) === '5608') console.log("path:" + modelId + ":" + path);
      if (path.length > 0) {
        var top = path[0];
        if (this.topLevelSynsets.has(top) && path.length > 1) {
          subSynsetCandidates.push(path[1]);
        }
      }
    }
  }
  //if (modelId.substr(0,4) === '5608') console.log("cand:" + modelId + ":" + subSynsetCandidates);
  var scope = this;
  if (subSynsetCandidates.length > 0) {  // pick largest subsynset
    subSynsetCandidates.sort(function(a,b) {
      var numA = scope.synsets[a].numInstances;
      var numB = scope.synsets[b].numInstances;
      if (numA < numB) return -1;
      if (numA > numB) return 1;
      return 0;
    });
    var topCandSynId = subSynsetCandidates[0];
    if (scope.synsets[topCandSynId].numInstances < scope.synsetMinNumInstances) {
      return model.topLevelSynsetId;  // too small so just use top-level synset
    } else {
      return topCandSynId;  // large enough to use
    }
  } else {
    return model.topLevelSynsetId;  // defalt if no other synset annotations
  }
};

// iterate over each model and write out modelId,topLevelSynsetId,subSynsetId
Taxonomy.prototype.printModelSynsets = function() {
  console.log("modelId,topLevelSynsetId,subSynsetId");
  for (var modelId in this.models) {
    if (!this.models.hasOwnProperty(modelId)) { continue; }
    var topId = this.models[modelId].topLevelSynsetId;
    var subId = this.findSubSynsetForModel(modelId);
    console.log(modelId + "," + topId + "," + subId);
  }
};

module.exports = Taxonomy;