require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'JumpVideoAnalysis'
  s.version        = package['version']
  s.summary        = package['description']
  s.description    = package['description']
  s.license        = package['license']
  s.author         = 'OpenAI'
  s.homepage       = 'https://example.invalid/jump-video-analysis'
  s.platforms      = {
    :ios => '15.1'
  }
  s.source         = { :git => 'https://example.invalid/jump-video-analysis.git' }
  s.static_framework = true
  s.swift_version  = '5.9'

  s.dependency 'ExpoModulesCore'
  s.frameworks = ['AVFoundation', 'Vision', 'ImageIO']
  s.source_files = '**/*.{h,m,swift}'
end
