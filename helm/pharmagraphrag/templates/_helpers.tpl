{{/*
Common labels applied to every resource.
*/}}
{{- define "pharmagraphrag.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end }}

{{/*
Selector labels for a given component.
Usage: {{ include "pharmagraphrag.selectorLabels" (dict "Chart" .Chart "Release" .Release "component" "api") }}
*/}}
{{- define "pharmagraphrag.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Component image string (registry/repository:tag).
Falls back to Chart.appVersion if no tag is set.
*/}}
{{- define "pharmagraphrag.image" -}}
{{- $componentTag := .componentImage.tag | default "" -}}
{{- $globalTag := .globalImage.tag | default "" -}}
{{- $tag := $componentTag | default $globalTag | default .chartAppVersion -}}
{{ .globalImage.registry }}/{{ .componentImage.repository }}:{{ $tag }}
{{- end }}

{{/*
Resolve the secret name (existing or generated).
*/}}
{{- define "pharmagraphrag.secretName" -}}
{{- if .Values.secrets.create -}}
{{ .Release.Name }}-secrets
{{- else -}}
{{ .Values.secrets.existingSecret }}
{{- end -}}
{{- end }}
