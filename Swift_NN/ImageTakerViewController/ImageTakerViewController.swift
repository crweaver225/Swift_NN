//
//  ImageTakerViewController.swift
//  Swift_NN
//
//  Created by Christopher Weaver on 1/28/19.
//  Copyright © 2019 Netronix. All rights reserved.
//

import UIKit
import AVFoundation
import MetalPerformanceShaders
import SwiftGifOrigin

class ImageTakerViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate  {
    
    var swiftNN : Swift_Neural_Network!
    var nnGif : UIImageView?

    override func viewDidLoad() {
        super.viewDidLoad()
        self.load_model()
        self.addPhoto()
    }
    
    func startDisplayingNNGif() {
        nnGif = UIImageView(frame: CGRect(x: self.view.frame.width * 0.25, y: self.view.frame.height * 0.4, width: self.view.frame.width * 0.5, height: self.view.frame.width * 0.5))
        nnGif?.image = UIImage.gif(name: "nn")
        self.view.addSubview(nnGif!)
    }

    func load_model() {
        if let path = Bundle.main.path(forResource: "mnist_model2", ofType: "json") {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
                let jsonResult = try JSONSerialization.jsonObject(with: data, options: .mutableLeaves) as! [String:AnyObject]
                swiftNN = Swift_Neural_Network()
                if let type = jsonResult["type"] as? [String] {
                    swiftNN.layers = swiftNN.convertToLayerTypes(layers: type)
                }
                if let activation_functions = jsonResult["activity_functions"] as? [String] {
                    swiftNN.activation_functions = swiftNN.convertToActivationFunctions(activationFunctions: activation_functions)
                }
                if let weights = jsonResult["weights"] as? [Matrix] {
                    swiftNN.weights = weights
                }
                if let filter = jsonResult["filters"] as? [Any] {
                    var filters : [[[Matrix]]] = []
                    for filter_data in filter {
                        if let fd = filter_data as? [[Matrix]] {
                            filters.append(fd)
                        } else {
                            filters.append([[[filter_data as! Array<Double>]]])
                        }
                    }
                    swiftNN.filters = filters
                }
                if let stride = jsonResult["stride"] as? [Int] {
                    swiftNN.stride = stride
                }
            } catch {
            }
        }
    }
    
    func addPhoto() {
        AVCaptureDevice.requestAccess(for: AVMediaType.video, completionHandler: { (granted :Bool) -> Void in
            if granted == true {
                DispatchQueue.main.async {
                    if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera) {
                        let imagePicker = UIImagePickerController()
                        imagePicker.delegate = self
                        imagePicker.sourceType = .camera
                        self.present(imagePicker, animated: true, completion: {
                            self.startDisplayingNNGif()
                        })
                    }
                }
            } else {
                let alertView = UIAlertController(title: "Camera could not be accessed", message: "Permission to use this device's camera has been denied. Go to settings to change permissions.", preferredStyle: .alert)
                alertView.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { action in
                }))
                DispatchQueue.main.async {
                    self.present(alertView, animated: true, completion: nil)
                }
            }
        });
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: {
        })
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        let info = convertFromUIImagePickerControllerInfoKeyDictionary(info)
        if let image = info[convertFromUIImagePickerControllerInfoKey(UIImagePickerController.InfoKey.originalImage)] as? UIImage {
            let orientationCorrectImage = image.fixOrientation()
            let invert = orientationCorrectImage.invertedImage()
            let grayScaleImage = self.convertToGrayScale(image: invert!)
            let resizedImage = self.resize2(with: grayScaleImage, scaledTo: CGSize(width: 28.0, height: 28.0))
            let pixelData = pixelValues(fromCGImage: resizedImage.cgImage)
            var arrayPixelData : [[Double]] = [[]]
            var arrayIndex : Int = 0
            for pixel in pixelData.pixelValues! {
                var modifiedPixel = 0
                if pixel > 225 {
                    modifiedPixel = Int(pixel)
                }
                let normalizedPixel = (Double(exactly: modifiedPixel) ?? 0.0) / 255.0
                
                if arrayPixelData[arrayIndex].count < 28 {
                    arrayPixelData[arrayIndex].append(normalizedPixel)
                } else {
                    arrayPixelData.append([normalizedPixel])
                    arrayIndex += 1
                }
            }
            picker.dismiss(animated: true, completion: {
                DispatchQueue.main.async {
                    if let results = self.swiftNN.executeNeuralNetwork(input: [arrayPixelData]) as? [[Double]] {
                        print(results)
                        self.pickNumber(results: results.first!)
                    }
                }
            })
        }
    }
    
    func pickNumber(results : [Double]) {
        var index = 0
        var choosenNumber = 0
        var currentNumber = 0.0
        for result in results {
            if result > currentNumber {
                choosenNumber = index
                currentNumber = result
            }
            index += 1
        }
        let alertView = UIAlertController(title: "Your number is:", message: "\(choosenNumber)", preferredStyle: .alert)
        alertView.addAction(UIAlertAction(title: "OK", style: .cancel, handler: {action in
            self.addPhoto()
        }))
        self.present(alertView, animated: true, completion: nil)
        self.nnGif?.removeFromSuperview()
    }
    
    func pixelValues(fromCGImage imageRef: CGImage?) -> (pixelValues: [UInt8]?, width: Int, height: Int){
        var width = 0
        var height = 0
        var pixelValues: [UInt8]?
        if let imageRef = imageRef {
            width = imageRef.width
            height = imageRef.height
            let bitsPerComponent = imageRef.bitsPerComponent
            let bytesPerRow = 28
            let totalBytes = height * bytesPerRow
            
            let colorSpace = CGColorSpaceCreateDeviceGray()
            var intensities = [UInt8](repeating: 0, count: totalBytes)
            
            let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
            contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
            
            pixelValues = intensities
        }
        
        return (pixelValues, width, height)
    }
    
    func resize2(with image: UIImage, scaledTo newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage ?? UIImage()
    }
    
    func convertToGrayScale(image: UIImage) -> UIImage {
        let imageRect:CGRect = CGRect(x:0, y:0, width:image.size.width, height: image.size.height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let width = image.size.width
        let height = image.size.height
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        let context = CGContext(data: nil, width: Int(width), height: Int(height), bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
        context?.draw(image.cgImage!, in: imageRect)
        let imageRef = context!.makeImage()
        let newImage = UIImage(cgImage: imageRef!)
        return newImage
    }
}

extension UIImage {
    func fixOrientation() -> UIImage {
        if self.imageOrientation == UIImage.Orientation.up {
            return self
        }
        UIGraphicsBeginImageContextWithOptions(self.size, false, self.scale)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        if let normalizedImage: UIImage = UIGraphicsGetImageFromCurrentImageContext() {
            UIGraphicsEndImageContext()
            return normalizedImage
        } else {
            return self
        }
    }
    
    func invertedImage() -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        let ciImage = CoreImage.CIImage(cgImage: cgImage)
        guard let filter = CIFilter(name: "CIColorInvert") else { return nil }
        filter.setDefaults()
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        let context = CIContext(options: nil)
        guard let outputImage = filter.outputImage else { return nil }
        guard let outputImageCopy = context.createCGImage(outputImage, from: outputImage.extent) else { return nil }
        return UIImage(cgImage: outputImageCopy)
    }
}

fileprivate func convertFromUIImagePickerControllerInfoKeyDictionary(_ input: [UIImagePickerController.InfoKey: Any]) -> [String: Any] {
    return Dictionary(uniqueKeysWithValues: input.map {key, value in (key.rawValue, value)})
}

fileprivate func convertFromUIImagePickerControllerInfoKey(_ input: UIImagePickerController.InfoKey) -> String {
    return input.rawValue
}



