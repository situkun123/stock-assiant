output "app_name" {
  description = "Koyeb app name"
  value       = koyeb_app.stock_assistant.name
}

output "service_url" {
  description = "Public URL of the deployed service"
  value       = "https://${var.app_name}-${var.koyeb_org}.koyeb.app"
}

output "koyeb_dashboard" {
  description = "Link to Koyeb dashboard"
  value       = "https://app.koyeb.com/apps/${var.app_name}"
}