// ---------------------------------------------------------------------------
// Medallion lakehouse landing infrastructure (project 14, ai-ml-portfolio)
//
// STATUS: compiled with the Bicep CLI (`bicep build`, output in main.json)
// but NEVER DEPLOYED - no Azure subscription was used for this project.
//
// Creates: ADLS Gen2 account (landing/bronze/silver/gold containers),
// Azure Data Factory with a system-assigned identity and data-plane RBAC on
// the storage account (no keys), Log Analytics for diagnostics, and an
// optional Microsoft Fabric capacity (F2) for the Lakehouse / notebooks.
// ---------------------------------------------------------------------------
targetScope = 'resourceGroup'

@description('Short prefix for resource names, e.g. "oqretail".')
@minLength(3)
@maxLength(11)
param prefix string

@description('Azure region.')
param location string = resourceGroup().location

@allowed(['dev', 'test', 'prod'])
param environment string = 'dev'

@description('Deploy a Fabric capacity (billed while running). Leave false to use a trial or existing capacity.')
param deployFabricCapacity bool = false

@description('Entra ID user principal name of the Fabric capacity administrator.')
param fabricAdminUpn string = ''

var tags = {
  project: 'medallion-lakehouse-portfolio'
  environment: environment
  dataClassification: 'public-dataset'
}
var storageName = toLower('${prefix}${environment}adls')
var containers = ['landing', 'bronze', 'silver', 'gold']
// Built-in role: Storage Blob Data Contributor
var blobDataContributor = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-${environment}-logs'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource adls 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    isHnsEnabled: true                // hierarchical namespace = ADLS Gen2
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false       // identities only
    supportsHttpsTrafficOnly: true
    accessTier: 'Hot'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: adls
  name: 'default'
  properties: {
    deleteRetentionPolicy: { enabled: true, days: 7 }
  }
}

resource zones 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [for c in containers: {
  parent: blobService
  name: c
  properties: { publicAccess: 'None' }
}]

resource adf 'Microsoft.DataFactory/factories@2018-06-01' = {
  name: '${prefix}-${environment}-adf'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

resource adfOnStorage 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(adls.id, adf.id, blobDataContributor)
  scope: adls
  properties: {
    roleDefinitionId: blobDataContributor
    principalId: adf.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource adlsLinkedService 'Microsoft.DataFactory/factories/linkedservices@2018-06-01' = {
  parent: adf
  name: 'ls_adls_lake'
  properties: {
    type: 'AzureBlobFS'
    typeProperties: {
      url: adls.properties.primaryEndpoints.dfs   // authenticates with the factory's managed identity
    }
  }
}

resource adfDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'to-log-analytics'
  scope: adf
  properties: {
    workspaceId: logs.id
    logs: [
      { category: 'PipelineRuns', enabled: true }
      { category: 'ActivityRuns', enabled: true }
    ]
  }
}

resource fabric 'Microsoft.Fabric/capacities@2023-11-01' = if (deployFabricCapacity) {
  name: toLower('${prefix}${environment}fabric')
  location: location
  tags: tags
  sku: { name: 'F2', tier: 'Fabric' }
  properties: {
    administration: { members: [fabricAdminUpn] }
  }
}

output storageAccountName string = adls.name
output dfsEndpoint string = adls.properties.primaryEndpoints.dfs
output dataFactoryName string = adf.name
output dataFactoryPrincipalId string = adf.identity.principalId
