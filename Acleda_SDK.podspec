Pod::Spec.new do |s|
  s.name             = 'Acleda_SDK'
  s.version          = '1.0.0'
  s.summary          = 'Acleda SDK with ONNX, MLKit, and Crypto support.'
  s.description      = 'Reusable SDK for face recognition and encryption using ONNX and GoogleMLKit.'
  s.homepage         = 'https://github.com/KongNoraksovann/Acleda_SDK'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Kong Noraksovann' => 'kongnoraksovann247@gmail.com' }
  s.platform         = :ios, '12.0'
  s.swift_versions   = ['5.0']
  s.source           = { :git => 'https://github.com/KongNoraksovann/Acleda_SDK.git', :tag => s.version.to_s }

  s.source_files     = 'Acleda_SDK/Source/**/*.{swift,h,m}'

  s.dependency 'onnxruntime-objc'
  s.dependency 'onnxruntime-c'
  s.dependency 'GoogleMLKit/FaceDetection'

  s.static_framework = false
end
